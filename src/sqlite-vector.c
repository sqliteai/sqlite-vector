//
//  sqlite-vector.c
//  sqlitevector
//
//  Created by Marco Bambini on 16/06/25.
//

#include "fp16/fp16.h"
#include "sqlite-vector.h"
#include "distance-cpu.h"

#include <math.h>
#include <float.h>
#include <stdio.h>
#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#ifndef _WIN32
#include <unistd.h>
#endif
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#if defined(_WIN32) || ((defined(__linux__) && !defined(__GLIBC__) && !defined(__ANDROID__))) || defined(SQLITE_WASM_EXTRA_INIT)
// Provide strcasestr function implementation for environments that lack it:
// - Windows (MinGW, MSVC, etc.)
// - Linux with non-glibc C libraries (musl, uclibc, etc.)
// - WebAssembly builds
char *strcasestr(const char *haystack, const char *needle) {
    if (!haystack || !needle) return NULL;
    if (!*needle) return (char *)haystack;
    for (; *haystack; ++haystack) {
        const char *h = haystack;
        const char *n = needle;
        while (*h && *n && tolower((unsigned char)*h) == tolower((unsigned char)*n)) {
            ++h;
            ++n;
        }
        if (!*n) return (char *)haystack;
    }
    return NULL;
}
#endif

#ifdef SQLITE_WASM_EXTRA_INIT
#define sqlite3_mutex_alloc(_type)                  NULL
#define sqlite3_mutex_enter(_mutex)
#define sqlite3_mutex_leave(_mutex)
#define sqlite3_mutex                               void
#endif

#ifndef SQLITE_CORE
SQLITE_EXTENSION_INIT1
#endif

#define DEBUG_VECTOR_ALWAYS(...)                    do {printf(__VA_ARGS__ );printf("\n");} while (0)

#if ENABLE_VECTOR_DEBUG
#define DEBUG_VECTOR(...)                           do {printf(__VA_ARGS__ );printf("\n");} while (0)
#else
#define DEBUG_VECTOR(...)
#endif

#define DEBUG_VECTOR_SERIALIZATION                  0
#if DEBUG_VECTOR_SERIALIZATION
#define VECTOR_PRINT(_b,_t,_n)                      vector_print(_b,_t,_n)
#else
#define VECTOR_PRINT(_b,_t,_n)
#endif

#define SKIP_SPACES(_p)                             while (*(_p) && isspace((unsigned char)*(_p))) (_p)++
#define TRIM_TRAILING(_start, _len)                 while ((_len) > 0 && isspace((unsigned char)(_start)[(_len) - 1])) (_len)--

#define DEFAULT_MAX_MEMORY                          30*1024*1024
#define MAX_TABLES                                  128
#define STATIC_SQL_SIZE                             2048

#define INT64_TO_INT8PTR(_val, _ptr)                do { \
                                                    (_ptr)[0] = (int8_t)(((_val) >> 0)  & 0xFF); \
                                                    (_ptr)[1] = (int8_t)(((_val) >> 8)  & 0xFF); \
                                                    (_ptr)[2] = (int8_t)(((_val) >> 16) & 0xFF); \
                                                    (_ptr)[3] = (int8_t)(((_val) >> 24) & 0xFF); \
                                                    (_ptr)[4] = (int8_t)(((_val) >> 32) & 0xFF); \
                                                    (_ptr)[5] = (int8_t)(((_val) >> 40) & 0xFF); \
                                                    (_ptr)[6] = (int8_t)(((_val) >> 48) & 0xFF); \
                                                    (_ptr)[7] = (int8_t)(((_val) >> 56) & 0xFF); \
                                                    } while(0)

#define INT64_FROM_INT8PTR(_ptr)                    \
                                                    ((int64_t)((uint8_t)(_ptr)[0]) | \
                                                    ((int64_t)((uint8_t)(_ptr)[1]) << 8) | \
                                                    ((int64_t)((uint8_t)(_ptr)[2]) << 16) | \
                                                    ((int64_t)((uint8_t)(_ptr)[3]) << 24) | \
                                                    ((int64_t)((uint8_t)(_ptr)[4]) << 32) | \
                                                    ((int64_t)((uint8_t)(_ptr)[5]) << 40) | \
                                                    ((int64_t)((uint8_t)(_ptr)[6]) << 48) | \
                                                    ((int64_t)((uint8_t)(_ptr)[7]) << 56))

#define SWAP(_t, a, b)                              do { _t tmp = (a); (a) = (b); (b) = tmp; } while (0)
#define KEY_MATCH(_k)                               (key_len == (int)sizeof(_k) - 1 && strncasecmp(key, _k, key_len) == 0)

#define VECTOR_COLUMN_IDX                           0
#define VECTOR_COLUMN_VECTOR                        1
#define VECTOR_COLUMN_K                             2
#define VECTOR_COLUMN_MEMIDX                        3
#define VECTOR_COLUMN_ROWID                         4
#define VECTOR_COLUMN_DISTANCE                      5

#define OPTION_KEY_TYPE                             "type"
#define OPTION_KEY_DIMENSION                        "dimension"
#define OPTION_KEY_NORMALIZED                       "normalized"
#define OPTION_KEY_MAXMEMORY                        "max_memory"
#define OPTION_KEY_DISTANCE                         "distance"
#define OPTION_KEY_QUANTTYPE                        "qtype"
#define OPTION_KEY_QUANTBITS                        "qbits"
#define OPTION_KEY_QUANTSCALE                       "qscale"        // used only in serialize/unserialize
#define OPTION_KEY_QUANTOFFSET                      "qoffset"       // used only in serialize/unserialize

#define VECTOR_INTERNAL_TABLE                       "CREATE TABLE IF NOT EXISTS _sqliteai_vector (tblname TEXT, colname TEXT, key TEXT, value ANY, PRIMARY KEY(tblname, colname, key));"

typedef struct turbo_rotation_plan turbo_rotation_plan;

// Reference-counted snapshot of the in-memory quantized index. A scan holds a reference
// for as long as it walks the buffer - streaming cursors keep it across many xNext calls -
// so vector_quantize_preload() and vector_quantize_cleanup() can replace or drop the
// table's copy without freeing memory a running scan is still reading.
typedef struct {
    int             refcount;               // guarded by qmutex
    int             counter;                // number of quantized vectors in data
    sqlite3_int64   bytes;                  // usable bytes in data
    uint8_t         data[];
} preload_index;

typedef struct {
    vector_type     v_type;                 // vector type
    int             v_dim;                  // vector dimension
    bool            v_normalized;           // is vector normalized ?
    vector_distance v_distance;             // vector distance function
    
    vector_qtype    q_type;                 // quantization type
    int             q_bits;                 // bit width for TurboQuant
    uint64_t        max_memory;             // max memory
} vector_options;

typedef struct {
    char            *t_name;                // table name
    char            *c_name;                // column name
    char            *pk_name;               // INTEGER primary key name (in case of WITHOUT ROWID tables) or rowid if NULL
    
    vector_options  options;                // options parsed in key=value arguments
    float           scale;                  // computed value by quantization
    float           offset;                 // computed value by quantization
    bool            binary_mean;            // binary mean option for 1BIT quantization
    
    preload_index   *preloaded;             // owned reference, guarded by qmutex

    turbo_rotation_plan *turbo_plan;
    int             turbo_plan_dim;
    bool            turbo_codebook_ready;
    int             turbo_codebook_dim;
    int             turbo_codebook_bits;
    float           turbo_boundaries[15];
    float           turbo_centroids[16];
} table_context;

typedef struct {
    table_context   tables[MAX_TABLES];     // simple array of MAX_TABLES tables
    int             table_count;            // number of entries in tables array
} vector_context;

typedef struct {
    sqlite3_vtab    base;                   // Base class - must be first
    sqlite3         *db;
    vector_context  *ctx;
} vFullScan;

typedef struct {
    sqlite3_vtab_cursor base;               // Base class - must be first
    table_context       *table;
    
    // STREAMING VT INTERFACE
    bool                is_streaming;
    bool                is_quantized;
    struct {
        int64_t             rowid;
        double              distance;
        distance_function_t distance_fn;
        
        sqlite3_stmt        *vm;
        void                *vector;
        int                 vsize;
        int                 vdim;
        
        void                *data;
        int                 dcounter;
        int                 dindex;
        int                 is_eof;
        float               turbo_qnorm_sq;
        int                 turbo_bits;
        sqlite3_int64       data_bytes;
        float               *turbo_query_lut;
        float               *turbo_norm_lut;
        int                 turbo_lut_rows;
        preload_index       *preload_ref;   // reference held while data points into it
    } stream;
    
    // NON-STREAMING VT INTERFACE
    int64_t             *rowids;
    double              *distance;
    int                 size;
    int                 row_index;
    int                 row_count;
} vFullScanCursor;

typedef bool (*keyvalue_callback)(sqlite3_context *context, void *xdata, const char *key, int key_len, const char *value, int value_len);
typedef int (*vcursor_run_callback)(sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size);
typedef int (*vcursor_sort_callback)(vFullScanCursor *c);

extern distance_function_t dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX];
extern const char *distance_backend_name;
extern const char *turbo_lut_backend_name;

static sqlite3_mutex *qmutex;

// MARK: - Preloaded Index -

static preload_index *preload_index_new (sqlite3_int64 bytes) {
    if (bytes <= 0) return NULL;
    preload_index *idx = (preload_index *)sqlite3_malloc64(sizeof(preload_index) + (sqlite3_uint64)bytes);
    if (!idx) return NULL;

    idx->refcount = 1;
    idx->counter = 0;
    idx->bytes = bytes;
    return idx;
}

static void preload_index_release (preload_index *idx) {
    if (!idx) return;

    sqlite3_mutex_enter(qmutex);
    int refs = --idx->refcount;
    sqlite3_mutex_leave(qmutex);

    if (refs == 0) sqlite3_free(idx);
}

// borrow the table's index for the duration of a scan (NULL if nothing is preloaded)
static preload_index *preload_index_acquire (table_context *t_ctx) {
    sqlite3_mutex_enter(qmutex);
    preload_index *idx = t_ctx->preloaded;
    if (idx) ++idx->refcount;
    sqlite3_mutex_leave(qmutex);

    return idx;
}

// hand a new index (or NULL) to the table and drop the reference to the previous one
static void preload_index_install (table_context *t_ctx, preload_index *idx) {
    sqlite3_mutex_enter(qmutex);
    preload_index *old = t_ctx->preloaded;
    t_ctx->preloaded = idx;
    sqlite3_mutex_leave(qmutex);

    preload_index_release(old);
}

// MARK: - SQLite Utils -

bool sqlite_system_exists (sqlite3 *db, const char *name, const char *type) {
    DEBUG_VECTOR("system_exists %s: %s", type, name);
    
    sqlite3_stmt *vm = NULL;
    bool result = false;
    
    char sql[1024];
    snprintf(sql, sizeof(sql), "SELECT EXISTS (SELECT 1 FROM sqlite_master WHERE type='%s' AND name=? COLLATE NOCASE);", type);
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto finalize;
    
    rc = sqlite3_bind_text(vm, 1, name, -1, SQLITE_STATIC);
    if (rc != SQLITE_OK) goto finalize;
    
    rc = sqlite3_step(vm);
    if (rc == SQLITE_ROW) {
        result = (bool)sqlite3_column_int(vm, 0);
        rc = SQLITE_OK;
    }
    
finalize:
    if (rc != SQLITE_OK) DEBUG_VECTOR_ALWAYS("Error executing %s in system_exists for type %s name %s (%s).", sql, type, name, sqlite3_errmsg(db));
    if (vm) sqlite3_finalize(vm);
    return result;
}

bool sqlite_table_exists (sqlite3 *db, const char *name) {
    return sqlite_system_exists(db, name, "table");
}

bool sqlite_trigger_exists (sqlite3 *db, const char *name) {
    return sqlite_system_exists(db, name, "trigger");
}

static bool context_result_error (sqlite3_context *context, int rc, const char *format, ...) {
    char buffer[4096];
    
    va_list arg;
    va_start (arg, format);
    vsnprintf(buffer, sizeof(buffer), format, arg);
    va_end (arg);
    
    if (context) {
        sqlite3_result_error(context, buffer, -1);
        sqlite3_result_error_code(context, rc);
    }
    
    return false;
}

static const char *sqlite_type_name (int type) {
    switch (type) {
        case SQLITE_TEXT: return "TEXT";
        case SQLITE_INTEGER: return "INTEGER";
        case SQLITE_FLOAT: return "REAL";
        case SQLITE_BLOB: return "BLOB";
    }
    return "N/A";
}

static char *sqlite_strdup (const char *str) {
    if (!str) return NULL;
    
    size_t len = strlen(str) + 1;
    char *result = (char*)sqlite3_malloc((int)len);
    if (result) memcpy(result, str, len);
    
    return result;
}

static void *sqlite_memdup (const void *v, int len) {
    if (!v) return NULL;
    
    void *result = (void *)sqlite3_malloc((int)len);
    if (result) memcpy(result, v, len);
    
    return result;
}

bool sqlite_column_exists (sqlite3 *db, const char *table_name, const char *column_name) {
    char sql[STATIC_SQL_SIZE];
    sqlite3_snprintf(sizeof(sql), sql, "SELECT EXISTS(SELECT 1 FROM pragma_table_info('%q') WHERE name = ?1);", table_name);
    bool result = false;
    
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, column_name, -1, SQLITE_STATIC);
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            result = (sqlite3_column_int(stmt, 0) != 0);
        }
    }
    
    sqlite3_finalize(stmt);
    return result;
}

bool sqlite_column_is_blob (sqlite3 *db, const char *table_name, const char *column_name) {
    char sql[STATIC_SQL_SIZE];
    sqlite3_snprintf(sizeof(sql), sql, "SELECT type FROM pragma_table_info('%q') WHERE name=?", table_name);
    bool result = false;
    
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, column_name, -1, SQLITE_STATIC);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            // see https://www.sqlite.org/datatype3.html (Determination Of Column Affinity)
            const char *type = (const char *)sqlite3_column_text(stmt, 0);
            result = (type == NULL) || strcasestr(type, "BLOB");
        }
    }
    
    sqlite3_finalize(stmt);
    return result;
}

static bool sqlite_table_is_without_rowid (sqlite3 *db, const char *table_name) {
    const char *sql = "SELECT sql FROM sqlite_master WHERE type='table' AND name=?";
    sqlite3_stmt *stmt = NULL;
    bool result = false;
    
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, table_name, -1, SQLITE_STATIC);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *statement = (const char *)sqlite3_column_text(stmt, 0);
            result = (statement && strcasestr(statement, "WITHOUT ROWID"));
        }
    }
    
    sqlite3_finalize(stmt);
    return result;
}

static char *sqlite_get_int_prikey_column (sqlite3 *db, const char *table_name) {
    char sql[STATIC_SQL_SIZE];
    // one row per PRIMARY KEY column. The statement takes no parameters: the old version
    // both bound one anyway and read bare type/name columns alongside COUNT(*), which
    // SQLite takes from an arbitrary row of the group.
    sqlite3_snprintf(sizeof(sql), sql, "SELECT name, type FROM pragma_table_info('%q') WHERE pk > 0;", table_name);
    char *prikey = NULL;

    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            const char *name = (const char *)sqlite3_column_text(stmt, 0);
            const char *decl_type = (const char *)sqlite3_column_text(stmt, 1);
            // see https://www.sqlite.org/datatype3.html (Determination Of Column Affinity)
            if (name && decl_type && strcasestr(decl_type, "INT")) prikey = sqlite_strdup(name);

            // a composite primary key gives more than one row and cannot stand in for the
            // rowid (step() invalidates name, so the copy has to be taken first)
            if (prikey && sqlite3_step(stmt) != SQLITE_DONE) {
                sqlite3_free(prikey);
                prikey = NULL;
            }
        }
    }

    sqlite3_finalize(stmt);
    return prikey;
}

static bool sqlite_sanity_check (sqlite3_context *context, const char *table_name, const char *column_name) {
    // sanity check table and column name
    sqlite3 *db = sqlite3_context_db_handle(context);
    
    // table_name must exists
    if (sqlite_table_exists(db, table_name) == false) {
        context_result_error(context, SQLITE_ERROR, "Table '%s' does not exist", table_name);
        return false;
    }
    
    // column_name must exists
    if (sqlite_column_exists(db, table_name, column_name) == false) {
        context_result_error(context, SQLITE_ERROR, "Column '%s' does not exist in table '%s'", column_name, table_name);
        return false;
    }
    
    // column_name must be of type BLOB
    if (sqlite_column_is_blob(db, table_name, column_name) == false) {
        context_result_error(context, SQLITE_ERROR, "Column '%s' in table '%s' must be of type BLOB", column_name, table_name);
        return false;
    }
    
    return true;
}

static int sqlite_vtab_set_error (sqlite3_vtab *vtab, const char *format, ...) {
    va_list arg;
    va_start (arg, format);
    char *err = sqlite3_vmprintf(format, arg);
    va_end (arg);
    
    vtab->zErrMsg = err;
    return SQLITE_ERROR;
}

static sqlite3_int64 sqlite_read_int64 (sqlite3 *db, const char *sql) {
    sqlite3_int64 value = 0;
    
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            value = sqlite3_column_int64(stmt, 0);
        }
    }
    
    sqlite3_finalize(stmt);
    return value;
}

static void *sqlite_common_set_error (sqlite3_context *context, sqlite3_vtab *vtab, int rc, const char *format, ...) {
    char buffer[4096];
    char *err = NULL;
    
    va_list arg;
    va_start (arg, format);
    if (vtab) err = sqlite3_vmprintf(format, arg);
    else if (context) vsnprintf(buffer, sizeof(buffer), format, arg);
    va_end (arg);
        
    if (vtab) {
        vtab->zErrMsg = err;
    } else if (context) {
        sqlite3_result_error(context, buffer, -1);
        sqlite3_result_error_code(context, rc);
    }
    
    return NULL;
}

static int sqlite_serialize (sqlite3_context *context, const char *table_name, const char *column_name, int type, const char *key, int64_t ivalue, double fvalue) {
    const char *sql = "REPLACE INTO _sqliteai_vector (tblname, colname, key, value) VALUES (?, ?, ?, ?);";
    sqlite3 *db = sqlite3_context_db_handle(context);
    sqlite3_stmt *vm = NULL;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_bind_text(vm, 1, table_name, -1, SQLITE_STATIC);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_bind_text(vm, 2, column_name, -1, SQLITE_STATIC);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_bind_text(vm, 3, key, -1, SQLITE_STATIC);
    if (rc != SQLITE_OK) goto cleanup;
    
    switch (type) {
        case SQLITE_INTEGER: rc = sqlite3_bind_int64(vm, 4, (sqlite3_int64)ivalue); break;
        case SQLITE_FLOAT: rc = sqlite3_bind_double(vm, 4, fvalue); break;
    }
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_step(vm);
    if (rc == SQLITE_DONE) rc = SQLITE_OK;
    
cleanup:
    if (rc != SQLITE_OK) sqlite3_result_error(context, sqlite3_errmsg(db), -1);
    if (vm) sqlite3_finalize(vm);
    return rc;
}

static int sqlite_unserialize (sqlite3_context *context, table_context *ctx) {
    const char *sql = "SELECT key, value FROM _sqliteai_vector WHERE tblname = ? AND colname = ?;";
    sqlite3 *db = sqlite3_context_db_handle(context);
    sqlite3_stmt *vm = NULL;
    
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_bind_text(vm, 1, ctx->t_name, -1, SQLITE_STATIC);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_bind_text(vm, 2, ctx->c_name, -1, SQLITE_STATIC);
    if (rc != SQLITE_OK) goto cleanup;
    
    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) {rc = SQLITE_OK; break;}
        if (rc != SQLITE_ROW) break;
        
        const char *key = (const char *)sqlite3_column_text(vm, 0);
        if (strcmp(key, OPTION_KEY_QUANTTYPE) == 0) {
            ctx->options.q_type = (vector_qtype)sqlite3_column_int(vm, 1);
            continue;
        }

        if (strcmp(key, OPTION_KEY_QUANTBITS) == 0) {
            ctx->options.q_bits = sqlite3_column_int(vm, 1);
            continue;
        }
        
        if (strcmp(key, OPTION_KEY_QUANTSCALE) == 0) {
            ctx->scale = (float)sqlite3_column_double(vm, 1);
            continue;
        }
        
        if (strcmp(key, OPTION_KEY_QUANTOFFSET) == 0) {
            ctx->offset = (float)sqlite3_column_double(vm, 1);
            continue;
        }
    }
    
cleanup:
    //if (rc != SQLITE_OK) sqlite3_result_error(context, sqlite3_errmsg(db), -1);
    if (vm) sqlite3_finalize(vm);
    return rc;
}

// MARK: - Quantization -

static inline uint8_t q_round_u8 (float s) {
    if (!isfinite(s)) {
        return (s > 0.0f) ? 255u : 0u;   /* NaN -> 0, +Inf -> 255, -Inf -> 0 */
    }
    float r = s + 0.5f * (1.0f - 2.0f * (s < 0.0f));  /* half away from zero */
    if (r >= 255.0f) return 255u;
    if (r <= 0.0f)   return 0u;
    int ir = (int)r;                      /* safe after the clamp above */
    return (uint8_t)ir;
}

static inline int8_t q_round_s8 (float s) {
    if (!isfinite(s)) {
        return (s > 0.0f) ? 127 : (s < 0.0f ? -128 : 0);
    }
    /* half-away-from-zero */
    float r = s + 0.5f * (1.0f - 2.0f * (s < 0.0f));
    if (r >= 127.0f)  return 127;
    if (r <= -128.0f) return -128;
    return (int8_t)(int)r;
}

// NOTE: these go through q_round_u8/q_round_s8 rather than casting first and clamping
// after. Converting a float to int is undefined when the value is NaN or outside the
// int range, and it does differ in practice: arm64 saturates, x86 yields INT_MIN. The
// helpers clamp in float and only then cast, which is also what every other quantizer
// below already did.
static inline void quantize_float32_to_unsigned8bit (const float *v, uint8_t *q, float offset, float scale, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        q[i]     = q_round_u8((v[i]     - offset) * scale);
        q[i + 1] = q_round_u8((v[i + 1] - offset) * scale);
        q[i + 2] = q_round_u8((v[i + 2] - offset) * scale);
        q[i + 3] = q_round_u8((v[i + 3] - offset) * scale);
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        q[i] = q_round_u8((v[i] - offset) * scale);
    }
}

static inline void quantize_float16_to_unsigned8bit (const uint16_t *v, uint8_t *q, float offset, float scale, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float x0 = float16_to_float32(v[i    ]);
        float x1 = float16_to_float32(v[i + 1]);
        float x2 = float16_to_float32(v[i + 2]);
        float x3 = float16_to_float32(v[i + 3]);

        q[i    ] = q_round_u8((x0 - offset) * scale);
        q[i + 1] = q_round_u8((x1 - offset) * scale);
        q[i + 2] = q_round_u8((x2 - offset) * scale);
        q[i + 3] = q_round_u8((x3 - offset) * scale);
    }
    for (; i < n; ++i) {
        float x = float16_to_float32(v[i]);
        q[i] = q_round_u8((x - offset) * scale);
    }
}

static inline void quantize_bfloat16_to_unsigned8bit (const uint16_t *v, uint8_t *q, float offset, float scale, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float x0 = bfloat16_to_float32(v[i    ]);
        float x1 = bfloat16_to_float32(v[i + 1]);
        float x2 = bfloat16_to_float32(v[i + 2]);
        float x3 = bfloat16_to_float32(v[i + 3]);

        q[i    ] = q_round_u8((x0 - offset) * scale);
        q[i + 1] = q_round_u8((x1 - offset) * scale);
        q[i + 2] = q_round_u8((x2 - offset) * scale);
        q[i + 3] = q_round_u8((x3 - offset) * scale);
    }
    for (; i < n; ++i) {
        float x = bfloat16_to_float32(v[i]);
        q[i] = q_round_u8((x - offset) * scale);
    }
}

static inline void quantize_u8_to_unsigned8bit (const uint8_t *v, uint8_t *q, float offset, float scale, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float x0 = (float)v[i    ];
        float x1 = (float)v[i + 1];
        float x2 = (float)v[i + 2];
        float x3 = (float)v[i + 3];

        q[i    ] = q_round_u8((x0 - offset) * scale);
        q[i + 1] = q_round_u8((x1 - offset) * scale);
        q[i + 2] = q_round_u8((x2 - offset) * scale);
        q[i + 3] = q_round_u8((x3 - offset) * scale);
    }
    for (; i < n; ++i) {
        float x = (float)v[i];
        q[i] = q_round_u8((x - offset) * scale);
    }
}

static inline void quantize_i8_to_unsigned8bit (const int8_t *v, uint8_t *q, float offset, float scale, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float x0 = (float)v[i    ];
        float x1 = (float)v[i + 1];
        float x2 = (float)v[i + 2];
        float x3 = (float)v[i + 3];

        q[i    ] = q_round_u8((x0 - offset) * scale);
        q[i + 1] = q_round_u8((x1 - offset) * scale);
        q[i + 2] = q_round_u8((x2 - offset) * scale);
        q[i + 3] = q_round_u8((x3 - offset) * scale);
    }
    for (; i < n; ++i) {
        float x = (float)v[i];
        q[i] = q_round_u8((x - offset) * scale);
    }
}

static inline void quantize_float32_to_signed8bit (const float *v, int8_t *q, float offset, float scale, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        q[i]     = q_round_s8((v[i]     - offset) * scale);
        q[i + 1] = q_round_s8((v[i + 1] - offset) * scale);
        q[i + 2] = q_round_s8((v[i + 2] - offset) * scale);
        q[i + 3] = q_round_s8((v[i + 3] - offset) * scale);
    }

    for (; i < n; ++i) {
        q[i] = q_round_s8((v[i] - offset) * scale);
    }
}

static inline void quantize_float16_to_signed8bit (const uint16_t *v, int8_t *q, float offset, float scale, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float x0 = float16_to_float32(v[i    ]);
        float x1 = float16_to_float32(v[i + 1]);
        float x2 = float16_to_float32(v[i + 2]);
        float x3 = float16_to_float32(v[i + 3]);

        q[i    ] = q_round_s8((x0 - offset) * scale);
        q[i + 1] = q_round_s8((x1 - offset) * scale);
        q[i + 2] = q_round_s8((x2 - offset) * scale);
        q[i + 3] = q_round_s8((x3 - offset) * scale);
    }
    for (; i < n; ++i) {
        float x = float16_to_float32(v[i]);
        q[i] = q_round_s8((x - offset) * scale);
    }
}

static inline void quantize_bfloat16_to_signed8bit (const uint16_t *v, int8_t *q, float offset, float scale, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float x0 = bfloat16_to_float32(v[i    ]);
        float x1 = bfloat16_to_float32(v[i + 1]);
        float x2 = bfloat16_to_float32(v[i + 2]);
        float x3 = bfloat16_to_float32(v[i + 3]);

        q[i    ] = q_round_s8((x0 - offset) * scale);
        q[i + 1] = q_round_s8((x1 - offset) * scale);
        q[i + 2] = q_round_s8((x2 - offset) * scale);
        q[i + 3] = q_round_s8((x3 - offset) * scale);
    }
    for (; i < n; ++i) {
        float x = bfloat16_to_float32(v[i]);
        q[i] = q_round_s8((x - offset) * scale);
    }
}

static inline void quantize_u8_to_signed8bit (const uint8_t *v, int8_t *q, float offset, float scale, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float x0 = (float)v[i    ];
        float x1 = (float)v[i + 1];
        float x2 = (float)v[i + 2];
        float x3 = (float)v[i + 3];

        q[i    ] = q_round_s8((x0 - offset) * scale);
        q[i + 1] = q_round_s8((x1 - offset) * scale);
        q[i + 2] = q_round_s8((x2 - offset) * scale);
        q[i + 3] = q_round_s8((x3 - offset) * scale);
    }
    for (; i < n; ++i) {
        float x = (float)v[i];
        q[i] = q_round_s8((x - offset) * scale);
    }
}

static inline void quantize_i8_to_signed8bit (const int8_t *v, int8_t *q, float offset, float scale, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float x0 = (float)v[i    ];
        float x1 = (float)v[i + 1];
        float x2 = (float)v[i + 2];
        float x3 = (float)v[i + 3];

        q[i    ] = q_round_s8((x0 - offset) * scale);
        q[i + 1] = q_round_s8((x1 - offset) * scale);
        q[i + 2] = q_round_s8((x2 - offset) * scale);
        q[i + 3] = q_round_s8((x3 - offset) * scale);
    }
    for (; i < n; ++i) {
        float x = (float)v[i];
        q[i] = q_round_s8((x - offset) * scale);
    }
}

static inline void quantize_float32 (const float *v, uint8_t *q, float offset, float scale, int dim, vector_qtype qtype) {
    if (qtype == VECTOR_QUANT_U8BIT) quantize_float32_to_unsigned8bit(v, q, offset, scale, dim);
    else quantize_float32_to_signed8bit(v, (int8_t *)q, offset, scale, dim);
}

static inline void quantize_float16 (const uint16_t *v, uint8_t *q, float offset, float scale, int dim, vector_qtype qtype) {
    if (qtype == VECTOR_QUANT_U8BIT) quantize_float16_to_unsigned8bit(v, q, offset, scale, dim);
    else quantize_float16_to_signed8bit(v, (int8_t *)q, offset, scale, dim);
}

static inline void quantize_bfloat16 (const uint16_t *v, uint8_t *q, float offset, float scale, int dim, vector_qtype qtype) {
    if (qtype == VECTOR_QUANT_U8BIT) quantize_bfloat16_to_unsigned8bit(v, q, offset, scale, dim);
    else quantize_bfloat16_to_signed8bit(v, (int8_t *)q, offset, scale, dim);
}

static inline void quantize_u8 (const uint8_t *v, uint8_t *q, float offset, float scale, int dim, vector_qtype qtype) {
    if (qtype == VECTOR_QUANT_U8BIT) quantize_u8_to_unsigned8bit(v, q, offset, scale, dim);
    else quantize_u8_to_signed8bit(v, (int8_t *)q, offset, scale, dim);
}

static inline void quantize_i8 (const int8_t *v, uint8_t *q, float offset, float scale, int dim, vector_qtype qtype) {
    if (qtype == VECTOR_QUANT_U8BIT) quantize_i8_to_unsigned8bit(v, q, offset, scale, dim);
    else quantize_i8_to_signed8bit(v, (int8_t *)q, offset, scale, dim);
}

static void quantize_binary (const float *input, uint8_t *output, int dim, bool is_binary_mean) {
      float threshold = 0.0f;

      if (is_binary_mean) {
          // compute mean as threshold
          float sum = 0.0f;
          for (int i = 0; i < dim; i++) {
              sum += input[i];
          }
          threshold = sum / dim;
      }

      // quantize
      memset(output, 0, (dim + 7) / 8);
      for (int i = 0; i < dim; i++) {
          if (input[i] >= threshold) {
              output[i / 8] |= (1 << (i % 8));
          }
      }
  }

static void quantize_binary_f16 (const uint16_t *input, uint8_t *output, int dim, bool is_binary_mean) {
    float threshold = 0.0f;

    if (is_binary_mean) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += float16_to_float32(input[i]);
        }
        threshold = sum / dim;
    }

    memset(output, 0, (dim + 7) / 8);
    for (int i = 0; i < dim; i++) {
        if (float16_to_float32(input[i]) >= threshold) {
            output[i / 8] |= (1 << (i % 8));
        }
    }
}

static void quantize_binary_bf16 (const uint16_t *input, uint8_t *output, int dim, bool is_binary_mean) {
    float threshold = 0.0f;

    if (is_binary_mean) {
        float sum = 0.0f;
        for (int i = 0; i < dim; i++) {
            sum += bfloat16_to_float32(input[i]);
        }
        threshold = sum / dim;
    }

    memset(output, 0, (dim + 7) / 8);
    for (int i = 0; i < dim; i++) {
        if (bfloat16_to_float32(input[i]) >= threshold) {
            output[i / 8] |= (1 << (i % 8));
        }
    }
}

static void quantize_binary_u8 (const uint8_t *input, uint8_t *output, int dim) {
    // For unsigned 8-bit, threshold at 128 (midpoint)
    memset(output, 0, (dim + 7) / 8);
    for (int i = 0; i < dim; i++) {
        if (input[i] >= 128) {
            output[i / 8] |= (1 << (i % 8));
        }
    }
}

static void quantize_binary_i8 (const int8_t *input, uint8_t *output, int dim) {
    // For signed 8-bit, threshold at 0 (sign-based)
    memset(output, 0, (dim + 7) / 8);
    for (int i = 0; i < dim; i++) {
        if (input[i] >= 0) {
            output[i / 8] |= (1 << (i % 8));
        }
    }
}

// MARK: - TurboQuant -

static inline size_t turbo_bytes_for_dim (int dim, int bits) {
    return ((size_t)dim * (size_t)bits + 7u) / 8u;
}

static inline uint64_t turbo_splitmix64 (uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

struct turbo_rotation_plan {
    int dim;
    int rounds;
    int pair_count;
    uint8_t *signs;
    int *a;
    int *b;
    float *s;
    float *c;
};

static void turbo_rotation_plan_free (turbo_rotation_plan *plan) {
    if (!plan) return;
    if (plan->signs) sqlite3_free(plan->signs);
    if (plan->a) sqlite3_free(plan->a);
    if (plan->b) sqlite3_free(plan->b);
    if (plan->s) sqlite3_free(plan->s);
    if (plan->c) sqlite3_free(plan->c);
    memset(plan, 0, sizeof(*plan));
}

static int turbo_rotation_plan_init (turbo_rotation_plan *plan, int dim) {
    memset(plan, 0, sizeof(*plan));
    plan->dim = dim;
    plan->rounds = 12;
    int pairs_per_round = dim / 2;
    plan->pair_count = pairs_per_round * plan->rounds;

    plan->signs = (uint8_t *)sqlite3_malloc64((sqlite3_uint64)dim);
    if (plan->pair_count > 0) {
        plan->a = (int *)sqlite3_malloc64((sqlite3_uint64)plan->pair_count * sizeof(int));
        plan->b = (int *)sqlite3_malloc64((sqlite3_uint64)plan->pair_count * sizeof(int));
        plan->s = (float *)sqlite3_malloc64((sqlite3_uint64)plan->pair_count * sizeof(float));
        plan->c = (float *)sqlite3_malloc64((sqlite3_uint64)plan->pair_count * sizeof(float));
    }
    if (!plan->signs || (plan->pair_count > 0 && (!plan->a || !plan->b || !plan->s || !plan->c))) {
        turbo_rotation_plan_free(plan);
        return SQLITE_NOMEM;
    }

    uint64_t sign_state = 0xA5A5A5A55A5A5A5Aull ^ (uint64_t)dim;
    for (int i = 0; i < dim; ++i) plan->signs[i] = (uint8_t)(turbo_splitmix64(&sign_state) & 1ull);

    int *perm = (int *)sqlite3_malloc64((sqlite3_uint64)dim * sizeof(int));
    if (!perm) {
        turbo_rotation_plan_free(plan);
        return SQLITE_NOMEM;
    }

    int idx = 0;
    for (int r = 0; r < plan->rounds; ++r) {
        for (int i = 0; i < dim; ++i) perm[i] = i;
        uint64_t state = 0xD1B54A32D192ED03ull ^ ((uint64_t)dim << 32) ^ (uint64_t)r;
        for (int i = dim - 1; i > 0; --i) {
            int j = (int)(turbo_splitmix64(&state) % (uint64_t)(i + 1));
            int tmp = perm[i];
            perm[i] = perm[j];
            perm[j] = tmp;
        }

        for (int i = 0; i + 1 < dim; i += 2) {
            uint64_t rnd = turbo_splitmix64(&state);
            float angle = (float)((double)(rnd >> 11) * (6.28318530717958647692 / 9007199254740992.0));
            plan->a[idx] = perm[i];
            plan->b[idx] = perm[i + 1];
            plan->s[idx] = sinf(angle);
            plan->c[idx] = cosf(angle);
            idx++;
        }
    }
    sqlite3_free(perm);
    return SQLITE_OK;
}

static float turbo_value_at (const void *v, vector_type type, int i) {
    switch (type) {
        case VECTOR_TYPE_F32: return ((const float *)v)[i];
        case VECTOR_TYPE_F16: return float16_to_float32(((const uint16_t *)v)[i]);
        case VECTOR_TYPE_BF16: return bfloat16_to_float32(((const uint16_t *)v)[i]);
        case VECTOR_TYPE_U8: return (float)((const uint8_t *)v)[i];
        case VECTOR_TYPE_I8: return (float)((const int8_t *)v)[i];
        case VECTOR_TYPE_BIT: return (float)((((const uint8_t *)v)[i / 8] >> (i % 8)) & 1);
    }
    return 0.0f;
}

static float turbo_copy_float (const void *v, vector_type type, int dim, float *out) {
    double norm_sq = 0.0;
    for (int i = 0; i < dim; ++i) {
        float x = turbo_value_at(v, type, i);
        out[i] = x;
        norm_sq += (double)x * (double)x;
    }
    return (float)norm_sq;
}

static void turbo_normalize_inplace (float *v, int dim, float norm_sq) {
    if (norm_sq <= 1e-20f) {
        memset(v, 0, (size_t)dim * sizeof(float));
        return;
    }
    float inv = 1.0f / sqrtf(norm_sq);
    for (int i = 0; i < dim; ++i) v[i] *= inv;
}

static void turbo_rotate_with_plan (const float *input, float *output, const turbo_rotation_plan *plan) {
    int dim = plan->dim;
    memcpy(output, input, (size_t)dim * sizeof(float));
    if (dim <= 1) return;

    for (int i = 0; i < dim; ++i) if (plan->signs[i]) output[i] = -output[i];

    for (int i = 0; i < plan->pair_count; ++i) {
        int a = plan->a[i];
        int b = plan->b[i];
        float s = plan->s[i];
        float c = plan->c[i];
        float x = output[a];
        float y = output[b];
        output[a] = c * x - s * y;
        output[b] = s * x + c * y;
    }
}

static inline double turbo_normal_pdf (double z) {
    return 0.39894228040143267794 * exp(-0.5 * z * z);
}

static inline double turbo_normal_cdf (double z) {
    return 0.5 * erfc(-z * 0.70710678118654752440);
}

static double turbo_beta_pdf_shifted (double x, int dim) {
    if (x <= -1.0 || x >= 1.0 || dim <= 1) return 0.0;
    double a = ((double)dim - 1.0) * 0.5;
    double y = (x + 1.0) * 0.5;
    double log_pdf = (a - 1.0) * (log(y) + log1p(-y)) + lgamma(2.0 * a) - 2.0 * lgamma(a) - log(2.0);
    return exp(log_pdf);
}

static void turbo_beta_interval_moments (double lo, double hi, int dim, double *prob_out, double *moment_out) {
    static const double nodes[16] = {
        0.048307665687738316,
        0.14447196158279649,
        0.23928736225213707,
        0.33186860228212767,
        0.42135127613063533,
        0.50689990893222939,
        0.58771575724076233,
        0.66304426693021520,
        0.73218211874028968,
        0.79448379596794241,
        0.84936761373256997,
        0.89632115576605212,
        0.93490607593773969,
        0.96476225558750643,
        0.98561151154526834,
        0.99726386184948156
    };
    static const double weights[16] = {
        0.09654008851472780,
        0.09563872007927486,
        0.09384439908080457,
        0.09117387869576388,
        0.08765209300440381,
        0.08331192422694676,
        0.07819389578707031,
        0.07234579410884851,
        0.06582222277636185,
        0.05868409347853555,
        0.05099805926237618,
        0.04283589802222668,
        0.03427386291302143,
        0.02539206530926206,
        0.01627439473090567,
        0.00701861000947010
    };

    if (lo < -1.0) lo = -1.0;
    if (hi > 1.0) hi = 1.0;
    if (hi <= lo) {
        *prob_out = 0.0;
        *moment_out = 0.0;
        return;
    }

    double mid = 0.5 * (lo + hi);
    double half = 0.5 * (hi - lo);
    double prob = 0.0;
    double moment = 0.0;
    for (int i = 0; i < 16; ++i) {
        double dx = half * nodes[i];
        double x1 = mid - dx;
        double x2 = mid + dx;
        double p1 = turbo_beta_pdf_shifted(x1, dim);
        double p2 = turbo_beta_pdf_shifted(x2, dim);
        prob += weights[i] * (p1 + p2);
        moment += weights[i] * (x1 * p1 + x2 * p2);
    }

    *prob_out = half * prob;
    *moment_out = half * moment;
}

static void turbo_make_codebook (int bits, int dim, float *boundaries, float *centroids) {
    int levels = 1 << bits;
    double c[16];
    double next[16];
    double sigma = (dim > 0) ? 1.0 / sqrt((double)dim) : 1.0;

    for (int i = 0; i < levels; ++i) {
        c[i] = (-3.0 + 6.0 * (double)i / (double)(levels - 1)) * sigma;
    }

    if (dim <= 1) {
        for (int it = 0; it < 80; ++it) {
            double max_change = 0.0;
            for (int i = 0; i < levels; ++i) {
                double lo = (i == 0) ? -INFINITY : 0.5 * (c[i - 1] + c[i]);
                double hi = (i == levels - 1) ? INFINITY : 0.5 * (c[i] + c[i + 1]);
                double zl = lo / sigma;
                double zh = hi / sigma;
                double p = turbo_normal_cdf(zh) - turbo_normal_cdf(zl);
                if (p <= 1e-15) {
                    next[i] = c[i];
                } else {
                    double pdf_lo = isinf(zl) ? 0.0 : turbo_normal_pdf(zl);
                    double pdf_hi = isinf(zh) ? 0.0 : turbo_normal_pdf(zh);
                    next[i] = sigma * (pdf_lo - pdf_hi) / p;
                }
                double change = fabs(next[i] - c[i]);
                if (change > max_change) max_change = change;
            }
            memcpy(c, next, (size_t)levels * sizeof(double));
            if (max_change < 1e-12) break;
        }

        for (int i = 0; i < levels - 1; ++i) boundaries[i] = (float)(0.5 * (c[i] + c[i + 1]));
        for (int i = 0; i < levels; ++i) centroids[i] = (float)c[i];
        return;
    }

    for (int it = 0; it < 200; ++it) {
        double max_change = 0.0;
        for (int i = 0; i < levels; ++i) {
            double lo = (i == 0) ? -1.0 : 0.5 * (c[i - 1] + c[i]);
            double hi = (i == levels - 1) ? 1.0 : 0.5 * (c[i] + c[i + 1]);
            double p = 0.0;
            double moment = 0.0;
            turbo_beta_interval_moments(lo, hi, dim, &p, &moment);
            if (p <= 1e-15) {
                next[i] = c[i];
            } else {
                next[i] = moment / p;
            }
            double change = fabs(next[i] - c[i]);
            if (change > max_change) max_change = change;
        }
        memcpy(c, next, (size_t)levels * sizeof(double));
        if (max_change < 1e-12) break;
    }

    for (int i = 0; i < levels - 1; ++i) boundaries[i] = (float)(0.5 * (c[i] + c[i + 1]));
    for (int i = 0; i < levels; ++i) centroids[i] = (float)c[i];
}

static inline uint8_t turbo_code_for_value (float x, const float *boundaries, int bits) {
    uint8_t code = 0;
    int nboundaries = (1 << bits) - 1;
    for (int i = 0; i < nboundaries; ++i) {
        if (x > boundaries[i]) ++code;
    }
    return code;
}

static void turbo_quantize_rotated (const float *rotated, uint8_t *packed, const float *boundaries, const float *centroids, int bits, int dim, float *inner_out) {
    memset(packed, 0, turbo_bytes_for_dim(dim, bits));

    double inner = 0.0;
    for (int j = 0; j < dim; ++j) {
        uint8_t code = turbo_code_for_value(rotated[j], boundaries, bits);
        inner += (double)rotated[j] * (double)centroids[code];
        size_t bit_pos = (size_t)j * (size_t)bits;
        size_t byte_pos = bit_pos / 8u;
        int shift = (int)(bit_pos % 8u);
        packed[byte_pos] |= (uint8_t)(code << shift);
        if (shift + bits > 8) packed[byte_pos + 1] |= (uint8_t)(code >> (8 - shift));
    }
    *inner_out = (float)inner;
}

static inline uint8_t turbo_unpack_code (const uint8_t *packed, int bits, int dim, int j) {
    (void)dim;
    size_t bit_pos = (size_t)j * (size_t)bits;
    size_t byte_pos = bit_pos / 8u;
    int shift = (int)(bit_pos % 8u);
    uint16_t value = packed[byte_pos];
    if (shift + bits > 8) value |= (uint16_t)packed[byte_pos + 1] << 8;
    return (uint8_t)((value >> shift) & ((1u << bits) - 1u));
}

static float turbo_distance_from_rotated_query (const float *query_rot, float query_norm_sq, const uint8_t *packed, float scale, const float *centroids, int bits, int dim, vector_distance distance) {
    double dot = 0.0;
    double xnorm_sq = 0.0;
    for (int j = 0; j < dim; ++j) {
        float c = centroids[turbo_unpack_code(packed, bits, dim, j)] * scale;
        dot += (double)query_rot[j] * (double)c;
        xnorm_sq += (double)c * (double)c;
    }

    switch (distance) {
        case VECTOR_DISTANCE_DOT:
            return (float)-dot;
        case VECTOR_DISTANCE_COSINE: {
            float d = (float)(1.0 - dot);
            return d < 0.0f ? 0.0f : d;
        }
        case VECTOR_DISTANCE_L2: {
            double d2 = (double)query_norm_sq + xnorm_sq - 2.0 * dot;
            if (d2 < 0.0) d2 = 0.0;
            return (float)sqrt(d2);
        }
        case VECTOR_DISTANCE_SQUARED_L2: {
            double d2 = (double)query_norm_sq + xnorm_sq - 2.0 * dot;
            return (float)(d2 < 0.0 ? 0.0 : d2);
        }
        default:
            return INFINITY;
    }
}

static float *turbo_build_query_lut (const float *query_rot, const float *centroids, int bits, int dim, int *lut_rows_out) {
    if (bits < 2 || bits > 4) {
        *lut_rows_out = 0;
        return NULL;
    }

    int codes_per_row = (bits == 3) ? 4 : (8 / bits);
    int entries_per_row = (bits == 3) ? 4096 : 256;
    int rows = (dim + codes_per_row - 1) / codes_per_row;
    float *lut = (float *)sqlite3_malloc64((sqlite3_uint64)rows * (sqlite3_uint64)entries_per_row * sizeof(float));
    if (!lut) {
        *lut_rows_out = 0;
        return NULL;
    }

    uint8_t mask = (uint8_t)((1u << bits) - 1u);
    for (int r = 0; r < rows; ++r) {
        for (int entry = 0; entry < entries_per_row; ++entry) {
            double sum = 0.0;
            for (int c = 0; c < codes_per_row; ++c) {
                int j = r * codes_per_row + c;
                if (j >= dim) break;
                int code = (entry >> (c * bits)) & mask;
                sum += (double)query_rot[j] * (double)centroids[code];
            }
            lut[(size_t)r * (size_t)entries_per_row + (size_t)entry] = (float)sum;
        }
    }

    *lut_rows_out = rows;
    return lut;
}

static float *turbo_build_norm_lut (const float *centroids, int bits, int dim, int *lut_rows_out) {
    if (bits < 2 || bits > 4) {
        *lut_rows_out = 0;
        return NULL;
    }

    int codes_per_row = (bits == 3) ? 4 : (8 / bits);
    int entries_per_row = (bits == 3) ? 4096 : 256;
    int rows = (dim + codes_per_row - 1) / codes_per_row;
    float *lut = (float *)sqlite3_malloc64((sqlite3_uint64)rows * (sqlite3_uint64)entries_per_row * sizeof(float));
    if (!lut) {
        *lut_rows_out = 0;
        return NULL;
    }

    uint8_t mask = (uint8_t)((1u << bits) - 1u);
    for (int r = 0; r < rows; ++r) {
        for (int entry = 0; entry < entries_per_row; ++entry) {
            double sum = 0.0;
            for (int c = 0; c < codes_per_row; ++c) {
                int j = r * codes_per_row + c;
                if (j >= dim) break;
                int code = (entry >> (c * bits)) & mask;
                double value = (double)centroids[code];
                sum += value * value;
            }
            lut[(size_t)r * (size_t)entries_per_row + (size_t)entry] = (float)sum;
        }
    }

    *lut_rows_out = rows;
    return lut;
}

// This used to carry a third copy of the lookup loop as a fallback for a null dispatch
// pointer, which init_distance_functions() always sets. One implementation is also what
// keeps every backend returning the same distance for the same query.
static inline float turbo_dot_from_lut (const uint8_t *packed, float scale, const float *query_lut, int lut_rows, int bits, int packed_bytes) {
    turbo_lut_dot_function_t fn = turbo_lut_dot_function;
    if (!fn) fn = turbo_lut_dot_cpu;
    return fn(packed, scale, query_lut, lut_rows, bits, packed_bytes);
}

static int table_context_ensure_turbo_plan (table_context *t_ctx, int dim) {
    if (t_ctx->turbo_plan && t_ctx->turbo_plan_dim == dim) return SQLITE_OK;

    if (t_ctx->turbo_plan) {
        turbo_rotation_plan_free(t_ctx->turbo_plan);
        sqlite3_free(t_ctx->turbo_plan);
        t_ctx->turbo_plan = NULL;
        t_ctx->turbo_plan_dim = 0;
    }

    turbo_rotation_plan *plan = (turbo_rotation_plan *)sqlite3_malloc64(sizeof(turbo_rotation_plan));
    if (!plan) return SQLITE_NOMEM;
    int rc = turbo_rotation_plan_init(plan, dim);
    if (rc != SQLITE_OK) {
        sqlite3_free(plan);
        return rc;
    }

    t_ctx->turbo_plan = plan;
    t_ctx->turbo_plan_dim = dim;
    return SQLITE_OK;
}

static int table_context_ensure_turbo_codebook (table_context *t_ctx, int bits, int dim) {
    if (bits < 2 || bits > 4) return SQLITE_MISUSE;
    if (t_ctx->turbo_codebook_ready && t_ctx->turbo_codebook_bits == bits && t_ctx->turbo_codebook_dim == dim) return SQLITE_OK;

    turbo_make_codebook(bits, dim, t_ctx->turbo_boundaries, t_ctx->turbo_centroids);
    t_ctx->turbo_codebook_ready = true;
    t_ctx->turbo_codebook_bits = bits;
    t_ctx->turbo_codebook_dim = dim;
    return SQLITE_OK;
}

static int table_context_require_turbo_cache (table_context *t_ctx, int bits, int dim) {
    if (!t_ctx->turbo_plan || t_ctx->turbo_plan_dim != dim) return SQLITE_MISUSE;
    if (!t_ctx->turbo_codebook_ready || t_ctx->turbo_codebook_bits != bits || t_ctx->turbo_codebook_dim != dim) return SQLITE_MISUSE;
    return SQLITE_OK;
}

static void table_context_free_turbo_cache (table_context *t_ctx) {
    if (t_ctx->turbo_plan) {
        turbo_rotation_plan_free(t_ctx->turbo_plan);
        sqlite3_free(t_ctx->turbo_plan);
    }
    t_ctx->turbo_plan = NULL;
    t_ctx->turbo_plan_dim = 0;
    t_ctx->turbo_codebook_ready = false;
    t_ctx->turbo_codebook_dim = 0;
    t_ctx->turbo_codebook_bits = 0;
}

// MARK: - General Utils -

static int vector_type_to_size (vector_type type) {
    switch (type) {
        case VECTOR_TYPE_F32:  return sizeof(float);        // 4 bytes
        case VECTOR_TYPE_F16:  return sizeof(uint16_t);     // 2 bytes
        case VECTOR_TYPE_BF16: return sizeof(uint16_t);     // 2 bytes
        case VECTOR_TYPE_U8:   return sizeof(uint8_t);      // 1 byte
        case VECTOR_TYPE_I8:   return sizeof(int8_t);       // 1 byte
        case VECTOR_TYPE_BIT:  return 0;                    // Special: use vector_bytes_for_dim()
    }
    return -1;                                              // error
}

static vector_type vector_name_to_type (const char *vname) {
    if ((strcasecmp(vname, "F32") == 0) || (strcasecmp(vname, "FLOAT32") == 0)) return VECTOR_TYPE_F32;
    if ((strcasecmp(vname, "F16") == 0) || (strcasecmp(vname, "FLOAT16") == 0)) return VECTOR_TYPE_F16;
    if ((strcasecmp(vname, "BF16") == 0) || (strcasecmp(vname, "FLOATB16") == 0)) return VECTOR_TYPE_BF16;
    if ((strcasecmp(vname, "U8") == 0) || (strcasecmp(vname, "UINT8") == 0)) return VECTOR_TYPE_U8;
    if ((strcasecmp(vname, "I8") == 0) || (strcasecmp(vname, "INT8") == 0)) return VECTOR_TYPE_I8;
    if (strcasecmp(vname, "BIT") == 0 || strcasecmp(vname, "BINARY") == 0 || strcasecmp(vname, "1BIT") == 0) return VECTOR_TYPE_BIT;
    return 0;
}

const char *vector_type_to_name (vector_type type) {
    switch (type) {
        case VECTOR_TYPE_F32: return "FLOAT32";
        case VECTOR_TYPE_F16: return "FLOAT16";
        case VECTOR_TYPE_BF16: return "FLOATB16";
        case VECTOR_TYPE_U8: return "UINT8";
        case VECTOR_TYPE_I8: return "INT8";
        case VECTOR_TYPE_BIT:  return "BIT";
    }
    return "N/A";
}

static size_t vector_bytes_for_dim (vector_type type, int dim) {
    // returns total bytes needed to store a vector of given type and dimension
    if (type == VECTOR_TYPE_BIT) {
        return (size_t)((dim + 7) / 8);  // Ceil division: pack 8 dimensions per byte
    }
    return (size_t)dim * vector_type_to_size(type);
}

static vector_qtype quant_name_to_type (const char *qname) {
    if (strcasecmp(qname, "UINT8") == 0) return VECTOR_QUANT_U8BIT;
    if (strcasecmp(qname, "INT8") == 0) return VECTOR_QUANT_S8BIT;
    if (strcasecmp(qname, "1BIT") == 0 || strcasecmp(qname, "BIT") == 0 || strcasecmp(qname, "BINARY") == 0) return VECTOR_QUANT_1BIT;
    if (strcasecmp(qname, "TURBO") == 0 || strcasecmp(qname, "TURBOQUANT") == 0 || strcasecmp(qname, "TURBO2") == 0 || strcasecmp(qname, "TURBO3") == 0 || strcasecmp(qname, "TURBO4") == 0) return VECTOR_QUANT_TURBO;
    return -1;
}

static vector_distance distance_name_to_type (const char *dname) {
    if (strcasecmp(dname, "L2") == 0) return VECTOR_DISTANCE_L2;
    if (strcasecmp(dname, "EUCLIDEAN") == 0) return VECTOR_DISTANCE_L2;
    if (strcasecmp(dname, "SQUARED_L2") == 0) return VECTOR_DISTANCE_SQUARED_L2;
    if (strcasecmp(dname, "COSINE") == 0) return VECTOR_DISTANCE_COSINE;
    if (strcasecmp(dname, "DOT") == 0) return VECTOR_DISTANCE_DOT;
    if (strcasecmp(dname, "INNER") == 0) return VECTOR_DISTANCE_DOT;
    if (strcasecmp(dname, "L1") == 0) return VECTOR_DISTANCE_L1;
    if (strcasecmp(dname, "MANHATTAN") == 0) return VECTOR_DISTANCE_L1;
    if (strcasecmp(dname, "HAMMING") == 0) return VECTOR_DISTANCE_HAMMING;
    return 0;
}

const char *vector_distance_to_name (vector_distance type) {
    switch (type) {
        case VECTOR_DISTANCE_L2: return "L2";
        case VECTOR_DISTANCE_SQUARED_L2: return "SQUARED_L2";
        case VECTOR_DISTANCE_COSINE: return "COSINE";
        case VECTOR_DISTANCE_DOT: return "DOT";
        case VECTOR_DISTANCE_L1: return "L1";
        case VECTOR_DISTANCE_HAMMING: return "HAMMING";
    }
    return "N/A";
}

// HAMMING is implemented for BIT vectors only, every other type implements
// everything but HAMMING. BIT columns are always scanned with HAMMING (the scan
// forces it), so any distance is accepted for them.
static bool vector_distance_is_supported (vector_distance distance, vector_type type) {
    if (type == VECTOR_TYPE_BIT) return true;
    return (distance != VECTOR_DISTANCE_HAMMING);
}

// bounds-checked lookup: returns NULL instead of an out-of-range or unpopulated
// entry, so a gap in the dispatch table becomes an error and not a crash
static distance_function_t vector_lookup_distance_function (vector_distance distance, vector_type type) {
    if ((int)distance <= 0 || (int)distance >= VECTOR_DISTANCE_MAX) return NULL;
    if ((int)type <= 0 || (int)type >= VECTOR_TYPE_MAX) return NULL;
    return dispatch_distance_table[distance][type];
}

// For unit-length vectors cosine distance is exactly 1 - dot, so the two norm
// accumulators - two thirds of the arithmetic in the cosine kernels - drop out of the
// inner loop. The DOT kernels return the negated dot product, hence the addition.
static float cosine_normalized_f32 (const void *v1, const void *v2, int n) {
    distance_function_t dot_fn = dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32];
    float d = 1.0f + dot_fn(v1, v2, n);
    if (d < 0.0f) return 0.0f;
    if (d > 2.0f) return 2.0f;
    return d;
}

// The stored vectors are unit length only because the caller said so with normalized=1,
// and only in full precision: the quantized index holds scaled integers whose norm is
// whatever the scale made it, so the shortcut is not valid there. The query is normalized
// once per scan in vCursorFilterCommon, which makes the result exact rather than merely
// rank-equivalent.
static bool vector_use_normalized_cosine (const table_context *t_ctx, bool quantized) {
    if (quantized) return false;
    if (!t_ctx->options.v_normalized) return false;
    if (t_ctx->options.v_distance != VECTOR_DISTANCE_COSINE) return false;
    if (t_ctx->options.v_type != VECTOR_TYPE_F32) return false;
    return (dispatch_distance_table[VECTOR_DISTANCE_DOT][VECTOR_TYPE_F32] != NULL);
}

#if DEBUG_VECTOR_SERIALIZATION
static void vector_print (void *buf, vector_type type, int n) {
    printf("type: %s - dim: %d [", vector_type_to_name(type), n);
    for (int i=0; i<n; ++i) {
        switch (type) {
            case VECTOR_TYPE_F32: {
                float *f = (float *)buf;
                printf("%f,", f[i]);
            }
            break;
                
            case VECTOR_TYPE_F16: {
                uint16_t *f = (uint16_t *)buf;
                printf("%f,", float16_to_float32(f[i]));
            }
            break;
                
            case VECTOR_TYPE_BF16: {
                uint16_t *f = (uint16_t *)buf;
                printf("%f,", bfloat16_to_float32(f[i]));
            }
            break;
                
            case VECTOR_TYPE_U8: {
                uint8_t *u = (uint8_t *)buf;
                printf("%d,", u[i]);
            }
            break;
                
            case VECTOR_TYPE_I8: {
                int8_t *u = (int8_t *)buf;
                printf("%d,", u[i]);
            }
            break;
                
            case VECTOR_TYPE_BIT: {
                uint8_t *b = (uint8_t *)buf;
                printf("%d,", (b[i / 8] >> (i % 8)) & 1);
            }
            break;
        }
    }
    printf("]\n");
}
#endif

static bool sanity_check_args (sqlite3_context *context, const char *func_name, int argc, sqlite3_value **argv, int ntypes, int *types) {
    if (argc != ntypes) {
        context_result_error(context, SQLITE_ERROR, "Function '%s' expects %d arguments, but %d were provided", func_name, ntypes, argc);
        return false;
    }
    
    for (int i=0; i<argc; ++i) {
        int actual_type = sqlite3_value_type(argv[i]);
        if (actual_type != types[i]) {
            context_result_error(context, SQLITE_ERROR, "Function '%s': argument %d must be of type %s (got %s)", func_name, (i+1), sqlite_type_name(types[i]), sqlite_type_name(actual_type));
            return false;
        }
    }
    
    return true;
}

static bool parse_keyvalue_string (sqlite3_context *context, const char *str, keyvalue_callback callback, void *xdata) {
    if (!str) return true;
    
    const char *p = str;
    while (*p) {
        SKIP_SPACES(p);
        
        const char *key_start = p;
        while (*p && *p != '=' && *p != ',') p++;
        
        int key_len = (int)(p - key_start);
        TRIM_TRAILING(key_start, key_len);
        
        if (*p != '=') {
            // Skip malformed pair
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
            continue;
        }
        
        p++; // skip '='
        SKIP_SPACES(p);
        
        const char *val_start = p;
        while (*p && *p != ',') p++;
        
        int val_len = (int)(p - val_start);
        TRIM_TRAILING(val_start, val_len);
        
        bool rc = callback(context, xdata, key_start, key_len, val_start, val_len);
        if (!rc) return rc;
        
        if (*p == ',') p++;
    }
    
    return true;
}

static uint64_t human_to_number (const char *s) {
    char *end = NULL;
    double d = strtod(s, &end);
    if ((d == 0) || (d == HUGE_VAL)) return 0;
    
    // skip whitespace before suffix
    SKIP_SPACES(end);
    
    // determine multiplier from suffix
    if (strncasecmp(end, "KB", 2) == 0) d *= 1024;
    else if (strncasecmp(end, "MB", 2) == 0) d *= 1024 * 1024;
    else if (strncasecmp(end, "GB", 2) == 0) d *= 1024 * 1024 * 1024;
    else if (*end != 0) return 0; // invalid suffix
    
    // sanity check
    if (d < 0 || d > (double)INT64_MAX) return 0;
    return (uint64_t)d;
}

bool vector_keyvalue_callback (sqlite3_context *context, void *xdata, const char *key, int key_len, const char *value, int value_len) {
    vector_options *options = (vector_options *)xdata;
    
    // sanity check
    if (!key || key_len == 0) return false;
    if (!value || value_len == 0) return false;
    
    // debug
    // printf("KEY: \"%.*s\", VALUE: \"%.*s\"\n", key_len, key, value_len, value);
    
    // convert value to c-string
    char buffer[256] = {0};
    size_t len = ((size_t)value_len > sizeof(buffer)-1) ? sizeof(buffer)-1 : (size_t)value_len;
    memcpy(buffer, value, len);

    if (KEY_MATCH(OPTION_KEY_TYPE)) {
        vector_type type = vector_name_to_type(buffer);
        if (type == 0) return context_result_error(context, SQLITE_ERROR, "Invalid vector type: '%s' is not a recognized type", buffer);
        options->v_type = type;
        return true;
    }
    
    if (KEY_MATCH(OPTION_KEY_DIMENSION)) {
        int dimension = (int)strtol(buffer, NULL, 0);
        if (dimension <= 0) return context_result_error(context, SQLITE_ERROR, "Invalid vector dimension: expected a positive integer, got '%s'", buffer);
        options->v_dim = dimension;
        return true;
    }
    
    if (KEY_MATCH(OPTION_KEY_NORMALIZED)) {
        int normalized = (int)strtol(buffer, NULL, 0);
        options->v_normalized = (normalized != 0);
        return true;
    }
    
    if (KEY_MATCH(OPTION_KEY_MAXMEMORY)) {
        uint64_t max_memory = human_to_number(buffer);
        if (max_memory > 0) options->max_memory = max_memory;
        return true;
    }
    
    if (KEY_MATCH(OPTION_KEY_QUANTTYPE)) {
        vector_qtype type = quant_name_to_type(buffer);
        if ((int)type == -1) return context_result_error(context, SQLITE_ERROR, "Invalid quantization type: '%s' is not a recognized or supported quantization type", buffer);
        options->q_type = type;
        if (type == VECTOR_QUANT_TURBO) {
            if (strcasecmp(buffer, "TURBO2") == 0) options->q_bits = 2;
            else if (strcasecmp(buffer, "TURBO3") == 0) options->q_bits = 3;
            else if (strcasecmp(buffer, "TURBO4") == 0) options->q_bits = 4;
            else if (options->q_bits == 0) options->q_bits = 4;
        }
        return true;
    }

    if (KEY_MATCH(OPTION_KEY_QUANTBITS)) {
        int bits = (int)strtol(buffer, NULL, 0);
        if (bits < 2 || bits > 4) return context_result_error(context, SQLITE_ERROR, "Invalid TurboQuant bit width: expected 2, 3, or 4, got '%s'", buffer);
        options->q_bits = bits;
        return true;
    }
    
    if (KEY_MATCH(OPTION_KEY_DISTANCE)) {
        vector_distance type = distance_name_to_type(buffer);
        if (type == 0) return context_result_error(context, SQLITE_ERROR, "Invalid distance name: '%s' is not a recognized or supported distance", buffer);
        options->v_distance = type;
        return true;
    }

    // means ignore unknown keys
    return true;
}

static inline int nearly_zero_float32 (float x) {
    return fabsf(x) <= 8.0f * FLT_EPSILON;  // tweak factor for your use
}

static inline size_t quantized_vector_bytes (vector_qtype qtype, int dim, int bits) {
    if (qtype == VECTOR_QUANT_1BIT) return (size_t)((dim + 7) / 8);
    if (qtype == VECTOR_QUANT_TURBO) return sizeof(float) + turbo_bytes_for_dim(dim, bits);
    return (size_t)dim * sizeof(uint8_t);
}

static inline size_t quantized_row_bytes (vector_qtype qtype, int dim, int bits) {
    return sizeof(int64_t) + quantized_vector_bytes(qtype, dim, bits);
}

// MARK: - SQL -

static char *generate_create_quant_table (const char *table_name, const char *column_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "CREATE TABLE IF NOT EXISTS \"vector0_%w_%w\" (rowid1 INTEGER, rowid2 INTEGER, counter INTEGER, data BLOB);", table_name, column_name);
}

static char *generate_drop_quant_table (const char *table_name, const char *column_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "DROP TABLE IF EXISTS \"vector0_%w_%w\";", table_name, column_name);
}

static char *generate_select_from_table (const char *table_name, const char *column_name, const char *pk_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "SELECT \"%w\", \"%w\" FROM \"%w\" ORDER BY \"%w\";", pk_name, column_name, table_name, pk_name);
}

static char *generate_select_quant_table (const char *table_name, const char *column_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "SELECT counter, data FROM \"vector0_%w_%w\";", table_name, column_name);
}

static char *generate_memory_quant_table (const char *table_name, const char *column_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "SELECT SUM(LENGTH(data)) FROM \"vector0_%w_%w\";", table_name, column_name);
}

static char *generate_insert_quant_table (const char *table_name, const char *column_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "INSERT INTO \"vector0_%w_%w\" (rowid1, rowid2, counter, data) VALUES (?, ?, ?, ?);", table_name, column_name);
}

static char *generate_quant_table_name (const char *table_name, const char *column_name, char sql[STATIC_SQL_SIZE]) {
    // NOTE: a plain name, not SQL - it is bound as a parameter, so no escaping here
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "vector0_%s_%s", table_name, column_name);
}

// MARK: - Vector Context and Options -

void *vector_context_create (void) {
    vector_context *ctx = (vector_context *)sqlite3_malloc(sizeof(vector_context));
    if (!ctx) return NULL;
    
    memset(ctx, 0, sizeof(vector_context));
    return (void *)ctx;
}

void vector_context_free (void *p) {
    if (p) {
        vector_context *ctx = (vector_context *)p;
        for (int i=0; i<ctx->table_count; ++i) {
            if (ctx->tables[i].t_name) sqlite3_free(ctx->tables[i].t_name);
            if (ctx->tables[i].c_name) sqlite3_free(ctx->tables[i].c_name);
            if (ctx->tables[i].pk_name) sqlite3_free(ctx->tables[i].pk_name);
            preload_index_release(ctx->tables[i].preloaded);
            table_context_free_turbo_cache(&ctx->tables[i]);
        }
        sqlite3_free(p);
    }
}

table_context *vector_context_lookup (vector_context *ctx, const char *table_name, const char *column_name) {
    if ((table_name == NULL) || (column_name == NULL)) return NULL;
    
    for (int i=0; i<ctx->table_count; ++i) {
        // tname and cname can be NULL after adding vector_cleanup function
        const char *tname = ctx->tables[i].t_name;
        const char *cname = ctx->tables[i].c_name;
        if (tname && cname && (strcasecmp(tname, table_name) == 0) && (strcasecmp(cname, column_name) == 0)) return &ctx->tables[i];
    }
    return NULL;
}

void vector_context_add (sqlite3_context *context, vector_context *ctx, const char *table_name, const char *column_name, vector_options *options) {
    // check if there is a free slot
    if (ctx->table_count >= MAX_TABLES) {
        context_result_error(context, SQLITE_ERROR, "Cannot add table: maximum number of allowed tables reached (%d)", MAX_TABLES);
        return;
    }
    
    char *t_name = sqlite_strdup(table_name);
    char *c_name = sqlite_strdup(column_name);
    if (!t_name || !c_name) {
        context_result_error(context, SQLITE_NOMEM, "Out of memory: unable to duplicate table or column name");
        if (t_name) sqlite3_free(t_name);
        if (c_name) sqlite3_free(c_name);
        return;
    }
    
    char *prikey = NULL;
    sqlite3 *db = sqlite3_context_db_handle(context);
    bool is_without_rowid = sqlite_table_is_without_rowid(db, table_name);
    prikey = (is_without_rowid == false) ? sqlite_strdup("rowid") : sqlite_get_int_prikey_column(db, table_name);

    // sanity check primary key
    if (!prikey) {
        (is_without_rowid) ? context_result_error(context, SQLITE_ERROR, "WITHOUT ROWID table '%s' must have exactly one PRIMARY KEY column of type INTEGER", table_name) : context_result_error(context, SQLITE_NOMEM, "Out of memory: unable to duplicate rowid column name");
        sqlite3_free(t_name);
        sqlite3_free(c_name);
        return;
    }
    
    int index = ctx->table_count;
    ctx->tables[index].t_name = t_name;
    ctx->tables[index].c_name = c_name;
    ctx->tables[index].pk_name = prikey;
    ctx->tables[index].options = *options;
    ctx->table_count++;
    
    int rc = sqlite_unserialize(context, &ctx->tables[index]);
    if (rc != SQLITE_OK) {
        context_result_error(context, rc, "Unable to load vector metadata for '%s.%s'", table_name, column_name);
        return;
    }
    if (ctx->tables[index].options.q_type == VECTOR_QUANT_TURBO) {
        int bits = ctx->tables[index].options.q_bits;
        int dim = ctx->tables[index].options.v_dim;
        rc = table_context_ensure_turbo_codebook(&ctx->tables[index], bits, dim);
        if (rc == SQLITE_OK) rc = table_context_ensure_turbo_plan(&ctx->tables[index], dim);
        if (rc != SQLITE_OK) context_result_error(context, rc, "Unable to initialize TurboQuant cache for '%s.%s'", table_name, column_name);
    }
}

void vector_options_init (vector_options *options) {
    memset(options, 0, sizeof(vector_options));
    options->v_type = VECTOR_TYPE_F32;
    options->v_distance = VECTOR_DISTANCE_L2;
    options->max_memory = DEFAULT_MAX_MEMORY;
    options->q_type = VECTOR_QUANT_AUTO;
    options->q_bits = 4;
}

vector_options vector_options_create (void) {
    vector_options options;
    vector_options_init(&options);
    return options;
}


// MARK: - Public -

static int vector_serialize_quantization (sqlite3 *db, const char *table_name, const char *column_name, uint32_t nrows, uint8_t *data, ptrdiff_t data_size, int64_t min_rowid, int64_t max_rowid) {
    
    char sql[STATIC_SQL_SIZE];
    generate_insert_quant_table(table_name, column_name, sql);
    
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto vector_serialize_quantization_cleanup;
    
    rc = sqlite3_bind_int64(vm, 1, min_rowid);
    if (rc != SQLITE_OK) goto vector_serialize_quantization_cleanup;
    
    rc = sqlite3_bind_int64(vm, 2, max_rowid);
    if (rc != SQLITE_OK) goto vector_serialize_quantization_cleanup;
    
    rc = sqlite3_bind_int(vm, 3, nrows);
    if (rc != SQLITE_OK) goto vector_serialize_quantization_cleanup;
    
    rc = sqlite3_bind_blob(vm, 4, (const void *)data, (int)data_size, SQLITE_STATIC);
    if (rc != SQLITE_OK) goto vector_serialize_quantization_cleanup;
    
    rc = sqlite3_step(vm);
    if (rc == SQLITE_DONE) rc = SQLITE_OK;
    
vector_serialize_quantization_cleanup:
    if (vm) sqlite3_finalize(vm);
    return rc;
}

static int vector_rebuild_quantization (sqlite3_context *context, const char *table_name, const char *column_name, table_context *t_ctx, vector_qtype qtype, int q_bits, uint64_t max_memory, uint32_t *count) {
    
    int rc = SQLITE_NOMEM;
    sqlite3_stmt *vm = NULL;
    char sql[STATIC_SQL_SIZE];
    sqlite3 *db = sqlite3_context_db_handle(context);
    uint32_t tot_processed = 0;
    
    const char *pk_name = t_ctx->pk_name;
    int dim = t_ctx->options.v_dim;
    vector_type type = t_ctx->options.v_type;
    float *turbo_values = NULL;
    float *turbo_rotated = NULL;
    
    if (qtype == VECTOR_QUANT_TURBO && (q_bits < 2 || q_bits > 4)) {
        context_result_error(context, SQLITE_ERROR, "TurboQuant requires qbits=2, 3, or 4");
        return SQLITE_MISUSE;
    }

    if (qtype == VECTOR_QUANT_TURBO && (type == VECTOR_TYPE_BIT || t_ctx->options.v_distance == VECTOR_DISTANCE_HAMMING || t_ctx->options.v_distance == VECTOR_DISTANCE_L1)) {
        context_result_error(context, SQLITE_ERROR, "TurboQuant supports FLOAT/INT vectors with DOT, COSINE, L2, or SQUARED_L2 distance");
        return SQLITE_MISUSE;
    }
    if (qtype != VECTOR_QUANT_TURBO) table_context_free_turbo_cache(t_ctx);

    // A BIT column is already binary, so 8-bit quantization has nothing to scale: it wrote
    // (dim+7)/8 bytes into a dim-byte slot and left the rest uninitialised. AUTO now means
    // the identity 1BIT, and an explicit 8-bit request is refused. This used to fail with
    // an unrelated message on a populated table and to silently record qtype=UINT8 on an
    // empty one, which then applied to rows inserted later.
    if (type == VECTOR_TYPE_BIT) {
        if (qtype == VECTOR_QUANT_AUTO) qtype = VECTOR_QUANT_1BIT;
        if (qtype != VECTOR_QUANT_1BIT) {
            context_result_error(context, SQLITE_ERROR, "BIT vectors can only be quantized with qtype=1BIT");
            return SQLITE_MISUSE;
        }
    }

    // compute size of a single quant, format is: rowid + quantized payload
    size_t q_size = quantized_row_bytes(qtype, dim, q_bits);
    if (q_size == 0) {
        sqlite3_result_error(context, "Vector dimension is zero, which is not possible", -1);
        return SQLITE_MISUSE;
    }
    
    // max_memory == 0 means use all required memory
    if (max_memory == 0) {
        sqlite3_snprintf(sizeof(sql), sql, "SELECT COUNT(*) FROM \"%w\";", table_name);
        int64_t count = sqlite_read_int64(db, sql);
        max_memory = (count == 0) ? DEFAULT_MAX_MEMORY : (uint64_t)count * (uint64_t)q_size;
        if (count <= 0) {
            // no vectors
            t_ctx->options.q_type = (qtype == VECTOR_QUANT_AUTO) ? VECTOR_QUANT_U8BIT : qtype;
            t_ctx->options.q_bits = q_bits;
            t_ctx->scale = 1.0f;
            t_ctx->offset = 0.0f;
            if (t_ctx->options.q_type == VECTOR_QUANT_TURBO) {
                rc = table_context_ensure_turbo_codebook(t_ctx, q_bits, dim);
                if (rc == SQLITE_OK) rc = table_context_ensure_turbo_plan(t_ctx, dim);
                if (rc != SQLITE_OK) {
                    context_result_error(context, rc, "Unable to initialize TurboQuant cache");
                    return rc;
                }
            }
            return SQLITE_OK;
        }
    }
    
    // max number of vectors that fits in max_memory (per batch; force at least 1)
    uint32_t max_vectors = (uint32_t)(max_memory / (uint64_t)q_size);
    if (max_vectors == 0) max_vectors = 1;
    
    sqlite3_uint64 out_bytes = (sqlite3_uint64)max_vectors * (sqlite3_uint64)q_size;
    uint8_t *data = sqlite3_malloc64(out_bytes);
    uint8_t *original = data;
    if (!data) goto vector_rebuild_quantization_cleanup;
        
    // SELECT rowid, embedding FROM table
    generate_select_from_table(table_name, column_name, pk_name, sql);
    rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto vector_rebuild_quantization_cleanup;
    
    // STEP 1
    // find global min/max across ALL vectors (skip for 1BIT quantization which uses fixed threshold)
    float min_val = FLT_MAX;
    float max_val = -FLT_MAX;
    bool contains_negative = false;

    if (qtype != VECTOR_QUANT_1BIT && qtype != VECTOR_QUANT_TURBO) {
        while (1) {
            rc = sqlite3_step(vm);
            if (rc == SQLITE_DONE) {rc = SQLITE_OK; break;}
            else if (rc != SQLITE_ROW) break;
            if (sqlite3_column_type(vm, 1) == SQLITE_NULL) continue;

            const void *blob = (float *)sqlite3_column_blob(vm, 1);
            if (!blob) continue;

            size_t blob_size = (size_t)sqlite3_column_bytes(vm, 1);
            size_t need_bytes = vector_bytes_for_dim(type, dim);
            if (blob_size < need_bytes) {
                context_result_error(context, SQLITE_ERROR, "Invalid vector blob found at rowid %lld", (long long)sqlite3_column_int64(vm, 0));
                rc = SQLITE_ERROR;
                goto vector_rebuild_quantization_cleanup;
            }

            for (int i = 0; i < dim; ++i) {
                float val = 0.0f;
                switch (type) {
                    case VECTOR_TYPE_F32:
                        val = ((float *)blob)[i];
                        break;
                    case VECTOR_TYPE_F16:
                        val = float16_to_float32(((uint16_t *)blob)[i]);
                        break;
                    case VECTOR_TYPE_BF16:
                        val = bfloat16_to_float32(((uint16_t *)blob)[i]);
                        break;
                    case VECTOR_TYPE_U8:
                        val = (float)(((uint8_t *)blob)[i]);
                        break;
                    case VECTOR_TYPE_I8:
                        val = (float)(((int8_t *)blob)[i]);
                        break;
                    default:
                        context_result_error(context, SQLITE_ERROR, "Unsupported vector type for 8-bit quantization");
                        rc = SQLITE_ERROR;
                        goto vector_rebuild_quantization_cleanup;
                }

                if (val < min_val) min_val = val;
                if (val > max_val) max_val = val;
                if (val < 0.0) contains_negative = true;
            }
        }
    }

    // set proper format
    if (qtype == VECTOR_QUANT_AUTO) {
        if (contains_negative == true) qtype = VECTOR_QUANT_S8BIT;
        else qtype = VECTOR_QUANT_U8BIT;
    }
    
    // STEP 2
    // compute scale and offset and set table them to table context standard min-max linear quantization
    float abs_max = fmaxf(fabsf(min_val), fabsf(max_val)); // only used in VECTOR_QUANT_S8BIT
    float range = max_val - min_val;
    float scale;
    if (qtype == VECTOR_QUANT_U8BIT) {
        scale = (range > 0.0f) ? (255.0f / range) : 1.0f;
    } else {
        scale = (abs_max > 0.0f) ? (127.0f / abs_max) : 1.0f;
    }
    // in the VECTOR_QUANT_S8BIT version I am assuming a symmetric quantization, for asymmetric quantization min_val should be used
    float offset = (qtype == VECTOR_QUANT_U8BIT) ? min_val : 0.0f;
    
    t_ctx->options.q_type = qtype;
    t_ctx->options.q_bits = q_bits;
    t_ctx->scale = scale;
    t_ctx->offset = offset;
    
    // restart processing from the beginning
    rc = sqlite3_reset(vm);
    if (rc != SQLITE_OK) goto vector_rebuild_quantization_cleanup;
    
    // STEP 3
    // actual quantization (ONLY 8bit is supported in this version)
    uint32_t n_processed = 0;
    int64_t min_rowid = 0, max_rowid = 0;
    if (qtype == VECTOR_QUANT_TURBO) {
        rc = table_context_ensure_turbo_codebook(t_ctx, q_bits, dim);
        if (rc != SQLITE_OK) goto vector_rebuild_quantization_cleanup;
        rc = table_context_ensure_turbo_plan(t_ctx, dim);
        if (rc != SQLITE_OK) goto vector_rebuild_quantization_cleanup;
        turbo_values = (float *)sqlite3_malloc64((sqlite3_uint64)dim * sizeof(float));
        turbo_rotated = (float *)sqlite3_malloc64((sqlite3_uint64)dim * sizeof(float));
        if (!turbo_values || !turbo_rotated) { rc = SQLITE_NOMEM; goto vector_rebuild_quantization_cleanup; }
    }
    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) {rc = SQLITE_OK; break;}
        else if (rc != SQLITE_ROW) break;
        if (sqlite3_column_type(vm, 1) == SQLITE_NULL) continue;
        
        int64_t rowid = (int64_t)sqlite3_column_int64(vm, 0);
        const void *blob = sqlite3_column_blob(vm, 1);
        if (!blob) continue;
        size_t blob_size = (size_t)sqlite3_column_bytes(vm, 1);
        size_t need_bytes = vector_bytes_for_dim(type, dim);
        if (blob_size < need_bytes) {
            context_result_error(context, SQLITE_ERROR, "Invalid vector blob found at rowid %lld", (long long)rowid);
            rc = SQLITE_ERROR;
            goto vector_rebuild_quantization_cleanup;
        }
        
        if (n_processed == 0) min_rowid = rowid;
        VECTOR_PRINT((void *)blob, type, dim);
        
        // copy rowid
        INT64_TO_INT8PTR(rowid, data);
        data += sizeof(int64_t);
        
        // quantize vector
        if (qtype == VECTOR_QUANT_TURBO) {
            float norm_sq = turbo_copy_float(blob, type, dim, turbo_values);
            turbo_normalize_inplace(turbo_values, dim, norm_sq);
            turbo_rotate_with_plan(turbo_values, turbo_rotated, t_ctx->turbo_plan);

            float inner = 0.0f;
            uint8_t *scale_ptr = data;
            data += sizeof(float);
            turbo_quantize_rotated(turbo_rotated, data, t_ctx->turbo_boundaries, t_ctx->turbo_centroids, q_bits, dim, &inner);

            float norm = sqrtf(norm_sq);
            float vector_scale = 0.0f;
            if (inner > 1e-10f) {
                vector_scale = (t_ctx->options.v_distance == VECTOR_DISTANCE_COSINE || t_ctx->options.v_normalized) ? (1.0f / inner) : (norm / inner);
            }
            memcpy(scale_ptr, &vector_scale, sizeof(float));
        } else if (qtype == VECTOR_QUANT_1BIT) {
            // 1-bit quantization: convert source to binary based on type
            switch (type) {
                case VECTOR_TYPE_F32: quantize_binary((const float *)blob, data, dim, t_ctx->binary_mean); break;
                case VECTOR_TYPE_F16: quantize_binary_f16((const uint16_t *)blob, data, dim, false); break;
                case VECTOR_TYPE_BF16: quantize_binary_bf16((const uint16_t *)blob, data, dim, false); break;
                case VECTOR_TYPE_U8: quantize_binary_u8((const uint8_t *)blob, data, dim); break;
                case VECTOR_TYPE_I8: quantize_binary_i8((const int8_t *)blob, data, dim); break;
                case VECTOR_TYPE_BIT: memcpy(data, blob, (dim + 7) / 8); break; // Already binary
            }
        } else {
            // 8-bit quantization (U8BIT or S8BIT)
            switch (type) {
                case VECTOR_TYPE_F32: quantize_float32((const float *)blob, data, offset, scale, dim, qtype); break;
                case VECTOR_TYPE_F16: quantize_float16((const uint16_t *)blob, data, offset, scale, dim, qtype); break;
                case VECTOR_TYPE_BF16: quantize_bfloat16((const uint16_t *)blob, data, offset, scale, dim, qtype); break;
                case VECTOR_TYPE_U8: quantize_u8((const uint8_t *)blob, data, offset, scale, dim, qtype); break;
                case VECTOR_TYPE_I8: quantize_i8((const int8_t *)blob, data, offset, scale, dim, qtype); break;
                // unreachable for new indexes (see the guard above), but one written by an
                // older build can still be loaded: zero the slot so the bytes past the
                // packed bits are never fed to a distance kernel uninitialised
                case VECTOR_TYPE_BIT: memset(data, 0, (size_t)dim); memcpy(data, blob, (dim + 7) / 8); break;
            }
        }
        
        #if DEBUG_VECTOR_SERIALIZATION
        vector_type qprint = (qtype == VECTOR_QUANT_1BIT) ? VECTOR_TYPE_BIT : (qtype == VECTOR_QUANT_U8BIT) ? VECTOR_TYPE_U8 : VECTOR_TYPE_I8;
        VECTOR_PRINT((void *)data, qprint, dim);
        #endif
        
        data += (qtype == VECTOR_QUANT_TURBO) ? turbo_bytes_for_dim(dim, q_bits) : ((qtype == VECTOR_QUANT_1BIT) ? ((dim + 7) / 8) : (dim * sizeof(uint8_t)));
        max_rowid = rowid;
        ++n_processed;
        ++tot_processed;
        
        if (n_processed == max_vectors) {
            size_t batch_size = data - original;  // compute actual bytes used
            rc = vector_serialize_quantization(db, table_name, column_name, n_processed, original, batch_size, min_rowid, max_rowid);
            if (rc != SQLITE_OK) goto vector_rebuild_quantization_cleanup;
            n_processed = 0;
            data = original;
        }
    }
    
    // handle remaining vectors
    if (n_processed > 0 && rc == SQLITE_OK) {
        size_t batch_size = data - original;
        rc = vector_serialize_quantization(db, table_name, column_name, n_processed, original, batch_size, min_rowid, max_rowid);
    }
    
vector_rebuild_quantization_cleanup:
    if (turbo_values) sqlite3_free(turbo_values);
    if (turbo_rotated) sqlite3_free(turbo_rotated);
    if (original) sqlite3_free(original);
    if (vm) sqlite3_finalize(vm);
    if (count) *count = tot_processed;
    return rc;
}

static void vector_quantize_preload (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT};
    if (sanity_check_args(context, "vector_quantize_preload", argc, argv, 2, types) == false) return;
    
    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);
    
    vector_context *v_ctx = (vector_context *)sqlite3_user_data(context);
    table_context *t_ctx = vector_context_lookup(v_ctx, table_name, column_name);
    if (!t_ctx) {
        context_result_error(context, SQLITE_ERROR, "Vector context not found for table '%s' and column '%s'. Ensure that vector_init() has been called before using vector_quantize_preload()", table_name, column_name);
        return;
    }
    
    // drop the previous preload: scans already walking it keep their own reference
    preload_index_install(t_ctx, NULL);

    if (t_ctx->options.q_type == VECTOR_QUANT_TURBO) {
        int rc = table_context_ensure_turbo_codebook(t_ctx, t_ctx->options.q_bits, t_ctx->options.v_dim);
        if (rc == SQLITE_OK) rc = table_context_ensure_turbo_plan(t_ctx, t_ctx->options.v_dim);
        if (rc != SQLITE_OK) {
            context_result_error(context, rc, "Unable to initialize TurboQuant cache");
            return;
        }
    }
    
    char sql[STATIC_SQL_SIZE];
    generate_memory_quant_table(table_name, column_name, sql);
    sqlite3 *db = sqlite3_context_db_handle(context);
    sqlite3_int64 required = sqlite_read_int64(db, sql);
    if (required <= 0) {
        context_result_error(context, SQLITE_ERROR, "Unable to read data from database. Ensure that vector_quantize() has been called before using vector_quantize_preload()");
        return;
    }
    
    preload_index *idx = preload_index_new(required);
    if (!idx) {
        context_result_error(context, SQLITE_NOMEM, "Out of memory: unable to allocate %lld bytes for quant buffer", (long long)required);
        return;
    }
    uint8_t *buffer = idx->data;
    
    sqlite3_stmt *vm = NULL;
    generate_select_quant_table(table_name, column_name, sql);
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) {
        context_result_error(context, rc, "Internal statement error: %s", sqlite3_errmsg(db));
        sqlite3_finalize(vm);
        preload_index_release(idx);
        return;
    }
    
    // the shadow table is an ordinary writable table and can change between the SUM that
    // sized the buffer and the scan below, so bound every copy against what was allocated
    const size_t row_stride = quantized_row_bytes(t_ctx->options.q_type, t_ctx->options.v_dim, t_ctx->options.q_bits);
    sqlite3_int64 seek = 0;
    sqlite3_int64 counter = 0;
    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) {rc = SQLITE_OK; break;} // return error: rebuild must be call (only if first time run)
        else if (rc != SQLITE_ROW) {break;}
        
        int n = sqlite3_column_int(vm, 0);
        int bytes = sqlite3_column_bytes(vm, 1);
        const uint8_t *data = (const uint8_t *)sqlite3_column_blob(vm, 1);
        
        if (!data || n < 0 || bytes < 0 || (sqlite3_int64)bytes > required - seek) {
            rc = SQLITE_CORRUPT;
            break;
        }
        
        memcpy(buffer + seek, data, (size_t)bytes);
        seek += bytes;
        counter += n;
    }
    sqlite3_finalize(vm);
    
    // the loaded rows must really hold the number of vectors they claim: this is the same
    // invariant the scans verify per chunk, checked once here so a malformed index is
    // rejected at preload time instead of at query time
    if ((rc == SQLITE_OK) && ((counter > INT_MAX) || ((sqlite3_uint64)seek < (sqlite3_uint64)counter * (sqlite3_uint64)row_stride))) {
        rc = SQLITE_CORRUPT;
    }
    
    if (rc != SQLITE_OK) {
        preload_index_release(idx);
        if (rc == SQLITE_CORRUPT) context_result_error(context, rc, "vector_quantize_preload failed: inconsistent quantization data for '%s.%s'", table_name, column_name);
        else context_result_error(context, rc, "vector_quantize_preload failed: %s", sqlite3_errmsg(db));
        return;
    }
    
    idx->counter = (int)counter;
    // seek, not required: rows removed between the two queries leave the tail of the
    // buffer uninitialised, and the scans bound themselves with this length
    idx->bytes = seek;
    preload_index_install(t_ctx, idx);
}

static int vector_quantize (sqlite3_context *context, const char *table_name, const char *column_name, const char *arg_options, bool *was_preloaded) {
    table_context *t_ctx = vector_context_lookup((vector_context *)sqlite3_user_data(context), table_name, column_name);
    if (!t_ctx) {
        context_result_error(context, SQLITE_ERROR, "Vector context not found for table '%s' and column '%s'. Ensure that vector_init() has been called before using vector_quantize()", table_name, column_name);
        return SQLITE_ERROR;
    }
    
    uint32_t counter = 0;
    int rc = SQLITE_ERROR;
    char sql[STATIC_SQL_SIZE];
    sqlite3 *db = sqlite3_context_db_handle(context);
    
    bool savepoint_open = false;
    rc = sqlite3_exec(db, "SAVEPOINT quantize;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    savepoint_open = true;
    
    generate_drop_quant_table(table_name, column_name, sql);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    
    generate_create_quant_table(table_name, column_name, sql);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    
    vector_options options = t_ctx->options; // t_ctx guarantees to exist
    bool res = parse_keyvalue_string(context, arg_options, vector_keyvalue_callback, &options);
    if (res == false) {rc = SQLITE_ERROR; goto quantize_cleanup;}
    
    sqlite3_mutex_enter(qmutex);
    rc = vector_rebuild_quantization(context, table_name, column_name, t_ctx, options.q_type, options.q_bits, options.max_memory, &counter);
    sqlite3_mutex_leave(qmutex);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    
    // serialize quantization options
    rc = sqlite_serialize(context, table_name, column_name, SQLITE_INTEGER, OPTION_KEY_QUANTTYPE, t_ctx->options.q_type, 0);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    rc = sqlite_serialize(context, table_name, column_name, SQLITE_INTEGER, OPTION_KEY_QUANTBITS, t_ctx->options.q_bits, 0);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    rc = sqlite_serialize(context, table_name, column_name, SQLITE_FLOAT, OPTION_KEY_QUANTSCALE, 0, t_ctx->scale);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    rc = sqlite_serialize(context, table_name, column_name, SQLITE_FLOAT, OPTION_KEY_QUANTOFFSET, 0, t_ctx->offset);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    
    rc = sqlite3_exec(db, "RELEASE quantize;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    savepoint_open = false;
    
    // success: returns the total number of quantized rows
    sqlite3_result_int64(context, (sqlite3_int64)counter);
    if (was_preloaded) {
        sqlite3_mutex_enter(qmutex);
        *was_preloaded = (t_ctx->preloaded != NULL);
        sqlite3_mutex_leave(qmutex);
    }
    return SQLITE_OK;
    
quantize_cleanup: {
        // capture before the rollback, which resets the connection's error state
        char *errmsg = (sqlite3_errcode(db) != SQLITE_OK) ? sqlite3_mprintf("%s", sqlite3_errmsg(db)) : NULL;
        if (savepoint_open) {
            sqlite3_exec(db, "ROLLBACK TO quantize;", NULL, NULL, NULL);
            sqlite3_exec(db, "RELEASE quantize;", NULL, NULL, NULL);
        }
        
        // only replace the message when SQLite actually has one: the callees set their own
        // through context_result_error, and overwriting it reported "not an error"
        if (errmsg) {
            sqlite3_result_error(context, errmsg, -1);
            sqlite3_free(errmsg);
        }
        sqlite3_result_error_code(context, rc);
        return rc;
    }
}

static void vector_quantize3 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT, SQLITE_TEXT};
    if (sanity_check_args(context, "vector_quantize", argc, argv, 3, types) == false) return;
    
    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);
    const char *options = (const char *)sqlite3_value_text(argv[2]);
    
    bool was_preloaded = false;
    int rc = vector_quantize(context, table_name, column_name, options, &was_preloaded);
    if ((rc == SQLITE_OK) && (was_preloaded)) vector_quantize_preload(context, 2, argv);
}

static void vector_quantize2 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT};
    if (sanity_check_args(context, "vector_quantize", argc, argv, 2, types) == false) return;
    
    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);
    
    bool was_preloaded = false;
    int rc = vector_quantize(context, table_name, column_name, NULL, &was_preloaded);
    if ((rc == SQLITE_OK) && (was_preloaded)) vector_quantize_preload(context, argc, argv);
}

static void vector_quantize_memory (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT};
    if (sanity_check_args(context, "vector_quantize_memory", argc, argv, 2, types) == false) return;
    
    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);
    
    char sql[STATIC_SQL_SIZE];
    generate_memory_quant_table(table_name, column_name, sql);
    
    sqlite3 *db = sqlite3_context_db_handle(context);
    sqlite3_int64 memory = sqlite_read_int64(db, sql);
    sqlite3_result_int64(context, memory);
}

static void vector_quantize_cleanup (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT};
    if (sanity_check_args(context, "vector_quantize_cleanup", argc, argv, 2, types) == false) return;

    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);

    vector_context *v_ctx = (vector_context *)sqlite3_user_data(context);
    table_context *t_ctx = vector_context_lookup(v_ctx, table_name, column_name);
    if (!t_ctx) return; // if no table context exists then do nothing

    // release any memory used in quantization: scans still walking the index keep it
    // alive through their own reference and free it when they are done
    preload_index_install(t_ctx, NULL);

    // drop quant table (if any)
    char sql[STATIC_SQL_SIZE];
    sqlite3 *db = sqlite3_context_db_handle(context);
    generate_drop_quant_table(table_name, column_name, sql);
    sqlite3_exec(db, sql, NULL, NULL, NULL);
}

// MARK: -

static void *vector_from_json (sqlite3_context *context, sqlite3_vtab *vtab, vector_type type, const char *json, int *size, int dimension) {
    char *blob = NULL;
    
    // skip leading whitespace
    SKIP_SPACES(json);
    
    // sanity check the JSON start array character
    if (*json != '[') {
        return sqlite_common_set_error(context, vtab, SQLITE_ERROR, "Malformed JSON: expected '[' at the beginning of the array");
    }
    json++;

    
    // count number of commas to estimate (over-estimate) the number of entries in the json
    int estimated_count = 0;
    for (const char *p = json; *p; ++p) {
        if (*p == ',') estimated_count++;
    }
    
    // allocate blob
    // For BIT type, each JSON element is a single bit, pack 8 per byte
    size_t item_size = vector_type_to_size(type);
    size_t alloc = (type == VECTOR_TYPE_BIT) ? ((estimated_count + 1) + 7) / 8 : (estimated_count + 1) * item_size;
    blob = sqlite3_malloc((int)alloc);
    if (!blob) {
        return sqlite_common_set_error(context, vtab, SQLITE_NOMEM, "Out of memory: unable to allocate %lld bytes for BLOB buffer", (long long)alloc);
    }
    if (type == VECTOR_TYPE_BIT) {
        memset(blob, 0, alloc);  // Initialize to zero for bit packing
    }

    // typed pointers
    float      *float_blob    = (float *)blob;
    uint8_t    *uint8_blob    = (uint8_t *)blob;
    int8_t     *int8_blob     = (int8_t *)blob;
    uint16_t   *uint16_blob   = (uint16_t *)blob;
    uint16_t   *bfloat16_blob = (uint16_t *)blob;
    
    int count = 0;
    const char *p = json;
    while (*p) {
        // skip whitespace
        SKIP_SPACES(p);
        
        // check for end-of-array character
        if (*p == ']') break;
        
        // parse number
        char *endptr;
        double value = strtod(p, &endptr);
        
        // sanity check
        if (p == endptr) {
            // parsing failed
            sqlite3_free(blob);
            return sqlite_common_set_error(context, vtab, SQLITE_ERROR, "Malformed JSON: expected a number at position %d (found '%c')", (int)(p - json) + 1, *p ? *p : '?');
        }
        
        // check bounds
        int max_count = (type == VECTOR_TYPE_BIT) ? (int)(alloc * 8) : (int)(alloc / item_size);
        if (count >= max_count) {
            sqlite3_free(blob);
            return sqlite_common_set_error(context, vtab, SQLITE_ERROR, "Too many elements in JSON array");
        }

        // convert to proper type
        switch (type) {
            case VECTOR_TYPE_F32:
                float_blob[count++] = (float)value;
                break;

            case VECTOR_TYPE_F16:
                uint16_blob[count++] = float32_to_float16((float)value);
                break;

            case VECTOR_TYPE_BF16:
                bfloat16_blob[count++] = float32_to_bfloat16((float)value);
                break;

            case VECTOR_TYPE_U8:
                if (value < 0 || value > 255) {
                    sqlite3_free(blob);
                    return sqlite_common_set_error(context, vtab, SQLITE_ERROR, "Value out of range for uint8_t");
                }
                uint8_blob[count++] = (uint8_t)value;
                break;

            case VECTOR_TYPE_I8:
                if (value < -128 || value > 127) {
                    sqlite3_free(blob);
                    return sqlite_common_set_error(context, vtab, SQLITE_ERROR, "Value out of range for int8_t");
                }
                int8_blob[count++] = (int8_t)value;
                break;

            case VECTOR_TYPE_BIT:
                if (value != 0 && value != 1) {
                    sqlite3_free(blob);
                    return sqlite_common_set_error(context, vtab, SQLITE_ERROR, "Value out of range for BIT: expected 0 or 1");
                }
                if ((int)value == 1) {
                    uint8_blob[count / 8] |= (1 << (count % 8));
                }
                count++;
                break;

            default:
                sqlite3_free(blob);
                return sqlite_common_set_error(context, vtab, SQLITE_ERROR, "Unsupported vector type");
        }
        
        p = endptr;
        
        // skip whitespace
        SKIP_SPACES(p);
        
        if (*p == ',') {
            // skip comma
            p++;
            
            // skip any whitespace after comma
            SKIP_SPACES(p);

            // allow trailing comma before closing ]
            if (*p == ']') break;
        } else if (*p == ']') {
            //end-of-array
            break;
        } else {
            sqlite3_free(blob);
            return sqlite_common_set_error(context, vtab, SQLITE_ERROR, "Malformed JSON: unexpected character '%c' at position %d", *p ? *p : '?', (int)(p - json) + 1);
        }
    }
    
    // sanity check vector dimension
    if ((dimension > 0) && (dimension != count)) {
        sqlite3_free(blob);
        return sqlite_common_set_error(context, vtab, SQLITE_ERROR, "Invalid JSON vector dimension: expected %d but found %d", dimension, count);
    }

    if (size) *size = (type == VECTOR_TYPE_BIT) ? (int)((count + 7) / 8) : (int)(count * item_size);
    return blob;
}

static void vector_as_type (sqlite3_context *context, vector_type type, int argc, sqlite3_value **argv) {
    sqlite3_value *value = argv[0];
    int value_size = sqlite3_value_bytes(value);
    int value_type = sqlite3_value_type(value);
    
    // dimension is an optional argument
    int dimension = (argc == 2) ? sqlite3_value_int(argv[1]) : 0;
    
    if (value_type == SQLITE_BLOB) {
        if (type == VECTOR_TYPE_BIT) {
            // For bit vectors, any size is valid (dimensions = size * 8, minus padding)
            // Optionally validate against expected dimension if provided
            if (dimension > 0) {
                size_t expected_size = vector_bytes_for_dim(type, dimension);
                if ((size_t)value_size != expected_size) {
                    context_result_error(context, SQLITE_ERROR,
                                         "Invalid BLOB size for format '%s': expected %d bytes for %d dimensions (got %d bytes)",
                                         vector_type_to_name(type), (int)expected_size, dimension, value_size);
                    return;
                }
            }
        } else {
            // the only check we can perform is that the blob size is an exact multiplier of the vector type
            if (value_size % vector_type_to_size(type) != 0) {
                context_result_error(context, SQLITE_ERROR,
                                     "Invalid BLOB size for format '%s': size must be a multiple of %d bytes",
                                     vector_type_to_name(type), vector_type_to_size(type));
                return;
            }
        }
        
        sqlite3_result_value(context, value);
        return;
    }
    
    if (value_type == SQLITE_TEXT) {
        // try to parse JSON array value
        const char *json = (const char *)sqlite3_value_text(value);
        if (!json) {
            context_result_error(context, SQLITE_ERROR, "Invalid TEXT input");
            return;
        }
        
        char *blob = vector_from_json(context, NULL, type, json, &value_size, dimension);
        if (!blob) return; // error is set in the context

        int print_dim = dimension;
        if (print_dim == 0) print_dim = (type == VECTOR_TYPE_BIT) ? value_size * 8 : value_size / vector_type_to_size(type);
        VECTOR_PRINT((void *)blob, type, print_dim);
        
        sqlite3_result_blob(context, (const void *)blob, value_size, sqlite3_free);
        return;
    }
    
    context_result_error(context, SQLITE_ERROR, "Unsupported input type: only BLOB and TEXT values are accepted (received %s)", sqlite_type_name(value_type));
}

static void vector_as_f32 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    vector_as_type(context, VECTOR_TYPE_F32, argc, argv);
}

static void vector_as_f16 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    vector_as_type(context, VECTOR_TYPE_F16, argc, argv);
}

static void vector_as_bf16 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    vector_as_type(context, VECTOR_TYPE_BF16, argc, argv);
}

static void vector_as_u8 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    vector_as_type(context, VECTOR_TYPE_U8, argc, argv);
}

static void vector_as_i8 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    vector_as_type(context, VECTOR_TYPE_I8, argc, argv);
}

static void vector_as_bit (sqlite3_context *context, int argc, sqlite3_value **argv) {
    vector_as_type(context, VECTOR_TYPE_BIT, argc, argv);
}

// MARK: - Modules -
static int vFullScanCursorNext (sqlite3_vtab_cursor *cur);
static int vStreamScanCursorRun (sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size);
static int vStreamQuantCursorRun (sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size);

static int vCursorFilterCommon (sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr, int argc, sqlite3_value **argv, const char *fname, vcursor_run_callback run_callback, vcursor_sort_callback sort_callback, vcursor_run_callback stream_callback, bool quantized) {

    vFullScanCursor *c = (vFullScanCursor *)cur;
    vFullScan *vtab = (vFullScan *)cur->pVtab;

    if (c->stream.vm) {
        sqlite3_finalize(c->stream.vm);
        c->stream.vm = NULL;
    }
    if (c->stream.vector) {
        sqlite3_free(c->stream.vector);
        c->stream.vector = NULL;
    }
    if (c->stream.turbo_query_lut) {
        sqlite3_free(c->stream.turbo_query_lut);
        c->stream.turbo_query_lut = NULL;
    }
    if (c->stream.turbo_norm_lut) {
        sqlite3_free(c->stream.turbo_norm_lut);
        c->stream.turbo_norm_lut = NULL;
    }
    if (c->stream.preload_ref) {
        preload_index_release(c->stream.preload_ref);
        c->stream.preload_ref = NULL;
    }
    memset(&c->stream, 0, sizeof(c->stream));

    if (argc != 3 && argc != 4) {
        return sqlite_vtab_set_error(&vtab->base, "%s expects 3 or 4 arguments, but %d were provided", fname, argc);
    }

    bool is_streaming = (argc == 3);
    bool is_quantized = quantized;
    c->is_streaming = is_streaming;
    c->is_quantized = is_quantized;
    
    // SQLITE_TEXT, SQLITE_TEXT, SQLITE_TEXT or SQLITE_BLOB, SQLITE_INTEGER
    for (int i=0; i<argc; ++i) {
        int actual_type = sqlite3_value_type(argv[i]);
        switch (i) {
            case 0:
            case 1:
                if (actual_type != SQLITE_TEXT)
                    return sqlite_vtab_set_error(&vtab->base, "%s: argument %d must be of type TEXT (got %s)", fname, (i+1), sqlite_type_name(actual_type));
                break;
            case 2:
                if ((actual_type != SQLITE_TEXT) && (actual_type != SQLITE_BLOB))
                    return sqlite_vtab_set_error(&vtab->base, "%s: argument %d must be of type TEXT or BLOB (got %s)", fname, (i+1), sqlite_type_name(actual_type));
                break;
            case 3:
                if (actual_type != SQLITE_INTEGER)
                    return sqlite_vtab_set_error(&vtab->base, "%s: argument %d must be of type INTEGER (got %s)", fname, (i+1), sqlite_type_name(actual_type));
                break;
        }
    }
    
    // retrieve arguments
    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);
    table_context *t_ctx = vector_context_lookup(vtab->ctx, table_name, column_name);
    if (!t_ctx) {
        return sqlite_vtab_set_error(&vtab->base, "%s: unable to retrieve context", fname);
    }
    
    const void *vector = NULL;
    bool vector_allocated = false;
    int vsize = 0;
    if (sqlite3_value_type(argv[2]) == SQLITE_TEXT) {
        vsize = sqlite3_value_bytes(argv[2]);
        vector = (const void *)vector_from_json(NULL, &vtab->base, t_ctx->options.v_type, (const char *)sqlite3_value_text(argv[2]), &vsize, t_ctx->options.v_dim);
        if (!vector) return SQLITE_ERROR; // error already set inside vector_from_json
        vector_allocated = true;
    } else {
        vector = (const void *)sqlite3_value_blob(argv[2]);
        vsize = sqlite3_value_bytes(argv[2]);
        if (!vector) return sqlite_vtab_set_error(&vtab->base, "%s: input vector cannot be NULL", fname);
        
        // the JSON branch above validates the dimension inside vector_from_json, the BLOB
        // branch must do it here: the distance functions read v_dim elements unconditionally
        size_t expected_bytes = vector_bytes_for_dim(t_ctx->options.v_type, t_ctx->options.v_dim);
        if ((size_t)vsize != expected_bytes) {
            return sqlite_vtab_set_error(&vtab->base, "%s: input vector must be %lld bytes (%d dimensions of type %s), but %d were provided", fname, (long long)expected_bytes, t_ctx->options.v_dim, vector_type_to_name(t_ctx->options.v_type), vsize);
        }
    }
    VECTOR_PRINT((void*)vector, t_ctx->options.v_type, t_ctx->options.v_dim);
    
    if (quantized) {
        char buffer[STATIC_SQL_SIZE];
        char *name = generate_quant_table_name(table_name, column_name, buffer);
        if (!name || !sqlite_table_exists(vtab->db, name)) {
            sqlite_vtab_set_error(&vtab->base, "Quantization table not found for table '%s' and column '%s'. Ensure that vector_quantize() has been called before using vector_quantize_scan()", table_name, column_name);
            if (vector_allocated) sqlite3_free((void *)vector);
            return SQLITE_ERROR;
        }
    }

    // with the 1 - dot shortcut the query has to be unit length too, and normalizing it
    // once per scan costs one pass over a single vector
    if (vector_use_normalized_cosine(t_ctx, quantized)) {
        int dim = t_ctx->options.v_dim;
        float *qn = (float *)sqlite_memdup(vector, vsize);
        if (!qn) {
            if (vector_allocated) sqlite3_free((void *)vector);
            return SQLITE_NOMEM;
        }
        double norm_sq = 0.0;
        for (int i = 0; i < dim; ++i) norm_sq += (double)qn[i] * (double)qn[i];
        if (norm_sq > 0.0) {
            float inv = (float)(1.0 / sqrt(norm_sq));
            for (int i = 0; i < dim; ++i) qn[i] *= inv;
        }
        if (vector_allocated) sqlite3_free((void *)vector);
        vector = qn;
        vector_allocated = true;
    }

    c->table = t_ctx;
    if (is_streaming) {
        int rc = stream_callback(vtab->db, c, vector, vsize);
        if (vector_allocated) sqlite3_free((void *)vector);
        if (rc != SQLITE_OK) return rc;
        return vFullScanCursorNext((sqlite3_vtab_cursor *)c);  // Position on first row
    }

    // non-streaming flow
    int k = sqlite3_value_int(argv[3]);
    if (k <= 0) {
        // an empty result, not an error: any non-OK return from xFilter is an error code
        // to SQLite, and SQLITE_DONE only looked harmless because it happens to be the
        // value sqlite3_step() reports at end of results
        if (vector_allocated) sqlite3_free((void *)vector);
        c->row_index = 0;
        c->row_count = 0;
        return SQLITE_OK;
    }

    if (c->row_count != k) {
        if (c->rowids) sqlite3_free(c->rowids);
        c->rowids = (int64_t *)sqlite3_malloc(k * sizeof(int64_t));
        if (c->rowids == NULL) {
            if (vector_allocated) sqlite3_free((void *)vector);
            return SQLITE_NOMEM;
        }

        if (c->distance) sqlite3_free(c->distance);
        c->distance = (double *)sqlite3_malloc(k * sizeof(double));
        if (c->distance == NULL) {
            if (vector_allocated) sqlite3_free((void *)vector);
            return SQLITE_NOMEM;
        }
    }

    memset(c->rowids, 0, k * sizeof(int64_t));
    for (int i=0; i<k; ++i) c->distance[i] = INFINITY;

    c->size = 0;
    c->row_index = 0;
    c->row_count = k;

    int rc = run_callback(vtab->db, c, vector, vsize);
    if (vector_allocated) sqlite3_free((void *)vector);
    if (rc != SQLITE_OK) return rc;
    int count = sort_callback(c);
    c->row_count -= count;

    #if 0
    for (int i=0; i<c->row_count; ++i) {
        printf("%lld\t%f\n", (long long)c->rowids[i], c->distance[i]);
    }
    #endif

    return rc;
}

static int vFullScanConnect (sqlite3 *db, void *pAux, int argc, const char *const *argv, sqlite3_vtab **ppVtab, char **pzErr) {
    // https://www.sqlite.org/vtab.html#table_valued_functions
    int rc = sqlite3_declare_vtab(db, "CREATE TABLE x(tbl hidden, vector hidden, k hidden, memidx hidden, id, distance);");
    if (rc != SQLITE_OK) return rc;
    
    vFullScan *vtab = (vFullScan *)sqlite3_malloc(sizeof(vFullScan));
    if (!vtab) return SQLITE_NOMEM;
    
    memset(vtab, 0, sizeof(vFullScan));
    vtab->db = db;
    vtab->ctx = (vector_context *)pAux;
    
    *ppVtab = (sqlite3_vtab *)vtab;
    return SQLITE_OK;
}

static int vFullScanDisconnect (sqlite3_vtab *pVtab) {
    vFullScan *vtab = (vFullScan *)pVtab;
    sqlite3_free(vtab);
    return SQLITE_OK;
}

static int vFullScanBestIndex (sqlite3_vtab *tab, sqlite3_index_info *pIdxInfo) {
    // With positional args to the table-valued function:
    //   3 args: f('tbl','col',vector)      → columns 0,1,2 constrained (streaming)
    //   4 args: f('tbl','col',vector,k)    → columns 0,1,2,3 constrained (top-k)
    // Column 2 (K) always receives the vector blob (positional arg 2).
    // Column 3 (MEMIDX) receives the actual k integer only with 4 args.
    // So top-k mode is determined by whether MEMIDX is constrained, not K.
    bool has_topk = false;

    const struct sqlite3_index_constraint *pConstraint = pIdxInfo->aConstraint;
    for(int i=0; i<pIdxInfo->nConstraint; i++, pConstraint++){
        if( pConstraint->usable == 0 ) continue;
        if( pConstraint->op != SQLITE_INDEX_CONSTRAINT_EQ ) continue;
        switch( pConstraint->iColumn ){
            case VECTOR_COLUMN_IDX:
                pIdxInfo->aConstraintUsage[i].argvIndex = 1;
                pIdxInfo->aConstraintUsage[i].omit = 1;
                break;
            case VECTOR_COLUMN_VECTOR:
                pIdxInfo->aConstraintUsage[i].argvIndex = 2;
                pIdxInfo->aConstraintUsage[i].omit = 1;
                break;
            case VECTOR_COLUMN_K:
                pIdxInfo->aConstraintUsage[i].argvIndex = 3;
                pIdxInfo->aConstraintUsage[i].omit = 1;
                break;
            case VECTOR_COLUMN_MEMIDX:
                has_topk = true;
                pIdxInfo->aConstraintUsage[i].argvIndex = 4;
                pIdxInfo->aConstraintUsage[i].omit = 1;
                break;
        }
    }

    if (has_topk) {
        // top-k mode: 4 positional args, argv[3] has the k integer
        pIdxInfo->estimatedCost = (double)1;
        pIdxInfo->estimatedRows = 100;
        pIdxInfo->idxNum = 1;
        
        // rows are emitted in ascending distance order, so that is the only ORDER BY we
        // may claim: telling SQLite otherwise makes it drop a sorter we do not replace
        pIdxInfo->orderByConsumed = (pIdxInfo->nOrderBy == 1 &&
                                     pIdxInfo->aOrderBy[0].iColumn == VECTOR_COLUMN_DISTANCE &&
                                     pIdxInfo->aOrderBy[0].desc == 0);
    } else {
        // streaming mode: 3 positional args, no sorting guaranteed
        pIdxInfo->estimatedCost = 1e8;
        pIdxInfo->estimatedRows = 100000;
        pIdxInfo->idxNum = 2;
    }

    return SQLITE_OK;
}

static int vFullScanCursorOpen (sqlite3_vtab *pVtab, sqlite3_vtab_cursor **ppCursor){
    vFullScanCursor *c = (vFullScanCursor *)sqlite3_malloc(sizeof(vFullScanCursor));
    if (!c) return SQLITE_NOMEM;
    
    memset(c, 0, sizeof(vFullScanCursor));
    *ppCursor = (sqlite3_vtab_cursor *)c;
    return SQLITE_OK;
}

static int vFullScanCursorClose (sqlite3_vtab_cursor *cur){
    vFullScanCursor *c = (vFullScanCursor *)cur;
    if (c->rowids) sqlite3_free(c->rowids);
    if (c->distance) sqlite3_free(c->distance);
    if (c->stream.vector) sqlite3_free(c->stream.vector);
    if (c->stream.turbo_query_lut) sqlite3_free(c->stream.turbo_query_lut);
    if (c->stream.turbo_norm_lut) sqlite3_free(c->stream.turbo_norm_lut);
    if (c->stream.vm) sqlite3_finalize(c->stream.vm);
    preload_index_release(c->stream.preload_ref);
    sqlite3_free(c);
    return SQLITE_OK;
}

static int vFullScanCursorNext (sqlite3_vtab_cursor *cur){
    vFullScanCursor *c = (vFullScanCursor *)cur;

    // non-streaming flow
    if (!c->is_streaming) { c->row_index++; return SQLITE_OK; }

    // streaming flow
    sqlite3_stmt *vm = c->stream.vm;
    void *v1 = c->stream.vector;
    int dimension = c->stream.vdim;
    distance_function_t distance_fn = c->stream.distance_fn;

    // FULL-SCAN
    if (!c->is_quantized) {
        // For BIT type, use byte count instead of dimension
        vector_type vt = c->table->options.v_type;
        int dist_size = (vt == VECTOR_TYPE_BIT) ? ((dimension + 7) / 8) : dimension;
        size_t expected_bytes = vector_bytes_for_dim(vt, dimension);
        while (1) {
            int rc = sqlite3_step(vm);
            if (rc == SQLITE_DONE) { c->stream.is_eof = 1; return SQLITE_OK; }
            else if (rc != SQLITE_ROW) return rc;

            // skip NULL values
            if (sqlite3_column_type(vm, 1) == SQLITE_NULL) continue;

            const float *v2 = (const float *)sqlite3_column_blob(vm, 1);
            if (v2 == NULL) continue;

            // skip undersized blobs
            if ((size_t)sqlite3_column_bytes(vm, 1) < expected_bytes) continue;

            float distance = distance_fn((const void *)v1, (const void *)v2, dist_size);
            if (nearly_zero_float32(distance)) distance = 0.0f;

            c->stream.distance = distance;
            c->stream.rowid = (int64_t)sqlite3_column_int64(vm, 0);
            return SQLITE_OK;
        }
    }

    if (c->table->options.q_type == VECTOR_QUANT_TURBO) {
        const size_t rowid_size = sizeof(int64_t);
        const size_t packed_size = turbo_bytes_for_dim(dimension, c->stream.turbo_bits);
        const size_t total_stride = rowid_size + sizeof(float) + packed_size;
        const float *centroids = c->table->turbo_centroids;
        vector_distance distance_type = c->table->options.v_distance;

        if (vm == NULL) {
            if (c->stream.data == NULL) return SQLITE_MISUSE;
            if (c->stream.dindex >= c->stream.dcounter) {
                c->stream.is_eof = 1;
                return SQLITE_OK;
            }
            if (c->stream.data_bytes < 0 || (sqlite3_uint64)c->stream.data_bytes < ((sqlite3_uint64)c->stream.dindex + 1u) * (sqlite3_uint64)total_stride) {
                return SQLITE_CORRUPT;
            }

            const uint8_t *current_data = (const uint8_t *)c->stream.data + ((size_t)c->stream.dindex * total_stride);
            float scale = 0.0f;
            memcpy(&scale, current_data + rowid_size, sizeof(float));
            const uint8_t *packed = current_data + rowid_size + sizeof(float);
            float distance;
            if (c->stream.turbo_query_lut && (distance_type == VECTOR_DISTANCE_DOT || distance_type == VECTOR_DISTANCE_COSINE)) {
                float dot = turbo_dot_from_lut(packed, scale, c->stream.turbo_query_lut, c->stream.turbo_lut_rows, c->stream.turbo_bits, (int)packed_size);
                distance = (distance_type == VECTOR_DISTANCE_DOT) ? -dot : (1.0f - dot);
                if (distance < 0.0f && distance_type == VECTOR_DISTANCE_COSINE) distance = 0.0f;
            } else if (c->stream.turbo_query_lut && c->stream.turbo_norm_lut && (distance_type == VECTOR_DISTANCE_L2 || distance_type == VECTOR_DISTANCE_SQUARED_L2)) {
                float dot = turbo_dot_from_lut(packed, scale, c->stream.turbo_query_lut, c->stream.turbo_lut_rows, c->stream.turbo_bits, (int)packed_size);
                float norm = turbo_dot_from_lut(packed, 1.0f, c->stream.turbo_norm_lut, c->stream.turbo_lut_rows, c->stream.turbo_bits, (int)packed_size);
                double d2 = (double)c->stream.turbo_qnorm_sq + ((double)scale * (double)scale * (double)norm) - 2.0 * (double)dot;
                if (d2 < 0.0) d2 = 0.0;
                distance = (distance_type == VECTOR_DISTANCE_L2) ? (float)sqrt(d2) : (float)d2;
            } else {
                distance = turbo_distance_from_rotated_query((const float *)v1, c->stream.turbo_qnorm_sq, packed, scale, centroids, c->stream.turbo_bits, dimension, distance_type);
            }
            if (nearly_zero_float32(distance)) distance = 0.0f;
            c->stream.distance = distance;
            c->stream.rowid = INT64_FROM_INT8PTR(current_data);
            c->stream.dindex++;
            return SQLITE_OK;
        }

        if (c->stream.dcounter == 0) {
            int rc = sqlite3_step(vm);
            if (rc == SQLITE_DONE) { c->stream.is_eof = 1; return SQLITE_OK; }
            else if (rc != SQLITE_ROW) return rc;

            c->stream.dcounter = sqlite3_column_int(vm, 0);
            c->stream.data = (uint8_t *)sqlite3_column_blob(vm, 1);
            c->stream.data_bytes = sqlite3_column_bytes(vm, 1);
            c->stream.dindex = 0;
            if (c->stream.data == NULL || c->stream.dcounter < 0 || c->stream.data_bytes < 0 || (sqlite3_uint64)c->stream.data_bytes < (sqlite3_uint64)c->stream.dcounter * (sqlite3_uint64)total_stride) {
                return SQLITE_CORRUPT;
            }
        }

        const uint8_t *current_data = (const uint8_t *)c->stream.data + ((size_t)c->stream.dindex * total_stride);
        float scale = 0.0f;
        memcpy(&scale, current_data + rowid_size, sizeof(float));
        const uint8_t *packed = current_data + rowid_size + sizeof(float);
        float distance;
        if (c->stream.turbo_query_lut && (distance_type == VECTOR_DISTANCE_DOT || distance_type == VECTOR_DISTANCE_COSINE)) {
            float dot = turbo_dot_from_lut(packed, scale, c->stream.turbo_query_lut, c->stream.turbo_lut_rows, c->stream.turbo_bits, (int)packed_size);
            distance = (distance_type == VECTOR_DISTANCE_DOT) ? -dot : (1.0f - dot);
            if (distance < 0.0f && distance_type == VECTOR_DISTANCE_COSINE) distance = 0.0f;
        } else if (c->stream.turbo_query_lut && c->stream.turbo_norm_lut && (distance_type == VECTOR_DISTANCE_L2 || distance_type == VECTOR_DISTANCE_SQUARED_L2)) {
            float dot = turbo_dot_from_lut(packed, scale, c->stream.turbo_query_lut, c->stream.turbo_lut_rows, c->stream.turbo_bits, (int)packed_size);
            float norm = turbo_dot_from_lut(packed, 1.0f, c->stream.turbo_norm_lut, c->stream.turbo_lut_rows, c->stream.turbo_bits, (int)packed_size);
            double d2 = (double)c->stream.turbo_qnorm_sq + ((double)scale * (double)scale * (double)norm) - 2.0 * (double)dot;
            if (d2 < 0.0) d2 = 0.0;
            distance = (distance_type == VECTOR_DISTANCE_L2) ? (float)sqrt(d2) : (float)d2;
        } else {
            distance = turbo_distance_from_rotated_query((const float *)v1, c->stream.turbo_qnorm_sq, packed, scale, centroids, c->stream.turbo_bits, dimension, distance_type);
        }
        if (nearly_zero_float32(distance)) distance = 0.0f;
        c->stream.distance = distance;
        c->stream.rowid = INT64_FROM_INT8PTR(current_data);
        c->stream.dindex++;

        if (c->stream.dindex == c->stream.dcounter) {
            c->stream.dcounter = 0;
            c->stream.data = NULL;
            c->stream.data_bytes = 0;
        }
        return SQLITE_OK;
    }

    // QUANTIZATION sizes
    const size_t rowid_size = sizeof(int64_t);
    const size_t vector_size = (size_t)c->stream.vsize;  // correctly set by caller for 1-bit or 8-bit
    const size_t total_stride = rowid_size + vector_size;

    // QUANTIZED IN-MEMORY
    if (vm == NULL) {
        if ((c->is_quantized == false) || (c->stream.data == NULL)) return SQLITE_MISUSE;

        // EOF if we've already consumed all items
        if (c->stream.dindex >= c->stream.dcounter) {
            c->stream.is_eof = 1;
            return SQLITE_OK;
        }
        if (c->stream.data_bytes < 0 || (sqlite3_uint64)c->stream.data_bytes < ((sqlite3_uint64)c->stream.dindex + 1u) * (sqlite3_uint64)total_stride) {
            return SQLITE_CORRUPT;
        }

        const uint8_t *data = (const uint8_t *)c->stream.data;
        size_t i = (size_t)c->stream.dindex;

        const uint8_t *current_data = data + (i * total_stride);
        const uint8_t *vector_data  = current_data + rowid_size;

        
        // no NULL vectors here by construction
        float distance = distance_fn((const void *)v1, (const void *)vector_data, c->stream.vsize);
        if (nearly_zero_float32(distance)) distance = 0.0f;

        c->stream.distance = distance;
        c->stream.rowid    = INT64_FROM_INT8PTR(current_data);
        c->stream.dindex++;
        return SQLITE_OK;
    }

    // QUANTIZED FROM DISK (chunked)
    if (c->stream.dcounter == 0) {
        int rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) { c->stream.is_eof = 1; return SQLITE_OK; }
        else if (rc != SQLITE_ROW) return rc;

        c->stream.dcounter = sqlite3_column_int(vm, 0);
        c->stream.data     = (uint8_t *)sqlite3_column_blob(vm, 1);
        c->stream.data_bytes = sqlite3_column_bytes(vm, 1);
        c->stream.dindex   = 0; // reset index for the new chunk
        if (c->stream.data == NULL || c->stream.dcounter < 0 || c->stream.data_bytes < 0 ||
            (sqlite3_uint64)c->stream.data_bytes < (sqlite3_uint64)c->stream.dcounter * (sqlite3_uint64)total_stride) {
            return SQLITE_CORRUPT;
        }
    }

    const uint8_t *data = (const uint8_t *)c->stream.data;
    size_t i = (size_t)c->stream.dindex;

    const uint8_t *current_data = data + (i * total_stride);
    const uint8_t *vector_data  = current_data + rowid_size;

    float distance = distance_fn((const void *)v1, (const void *)vector_data, c->stream.vsize);
    if (nearly_zero_float32(distance)) distance = 0.0f;

    c->stream.distance = distance;
    c->stream.rowid    = INT64_FROM_INT8PTR(current_data);
    c->stream.dindex++;

    if (c->stream.dindex == c->stream.dcounter) {
        // finished current chunk; force reload on next call
        c->stream.dcounter = 0;
        c->stream.data = NULL; // clear stale pointer to blob memory
        c->stream.data_bytes = 0;
    }

    return SQLITE_OK;
}


static int vFullScanCursorEof (sqlite3_vtab_cursor *cur){
    vFullScanCursor *c = (vFullScanCursor *)cur;
    return (c->is_streaming) ? c->stream.is_eof : (c->row_index == c->row_count);
}

static int vFullScanCursorColumn (sqlite3_vtab_cursor *cur, sqlite3_context *context, int iCol) {
    vFullScanCursor *c = (vFullScanCursor *)cur;
    if (iCol == VECTOR_COLUMN_ROWID) {
        sqlite3_result_int64(context, (c->is_streaming) ? (sqlite3_int64)c->stream.rowid : (sqlite3_int64)c->rowids[c->row_index]);
    } else if (iCol == VECTOR_COLUMN_DISTANCE) {
        sqlite3_result_double(context, (c->is_streaming) ? c->stream.distance : c->distance[c->row_index]);
    }
    return SQLITE_OK;
}

static int vFullScanCursorRowid (sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
    vFullScanCursor *c = (vFullScanCursor *)cur;
    *pRowid = (c->is_streaming) ? (sqlite3_int64)c->stream.rowid : (sqlite_int64)c->rowids[c->row_index];
    return SQLITE_OK;
}

// The candidate set is a binary max-heap over (distance, rowid), kept in the parallel
// arrays the cursor already owns. The root is the worst entry still in the set, so
// admitting a candidate costs one comparison and O(log k) to reinsert - the previous
// version rescanned all k slots for a new maximum on every improvement, and then paid a
// second O(k^2) to sort. It also removes a stale-index hazard: the old max_index was
// never reset between filters, so a cursor reused with a smaller k read past the end of
// the reallocated array. The root of a heap is always slot zero.
static inline void vTopKSiftDown (double *distance, int64_t *rowids, int n, int root) {
    for (;;) {
        int child = 2 * root + 1;
        if (child >= n) break;
        if (child + 1 < n && distance[child + 1] > distance[child]) ++child;
        if (distance[child] <= distance[root]) break;

        SWAP(double, distance[root], distance[child]);
        SWAP(int64_t, rowids[root], rowids[child]);
        root = child;
    }
}

// evicts the worst entry in favour of a better one; callers check distance[0] first
static inline void vTopKReplaceWorst (vFullScanCursor *c, double distance, int64_t rowid) {
    c->distance[0] = distance;
    c->rowids[0] = rowid;
    vTopKSiftDown(c->distance, c->rowids, c->row_count, 0);
}

// Heapsort in place: repeatedly move the largest entry past the end of the heap, which
// leaves the array ascending - the order the cursor emits rows in. Returns how many
// trailing slots were never filled so the caller can trim them.
static int vFullScanSortSlots (vFullScanCursor *c) {
    double  *distance = c->distance;
    int64_t *rowids = c->rowids;
    int     n = c->row_count;

    for (int end = n - 1; end > 0; --end) {
        SWAP(double, distance[0], distance[end]);
        SWAP(int64_t, rowids[0], rowids[end]);
        vTopKSiftDown(distance, rowids, end, 0);
    }

    int counter = 0;
    while (counter < n && distance[n - 1 - counter] == INFINITY) ++counter;
    return counter;
}

static int vFullScanRun (sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size) {
    const char *pk_name = c->table->pk_name;
    const char *col_name = c->table->c_name;
    const char *table_name = c->table->t_name;
    int dimension = c->table->options.v_dim;
    
    char *sql = sqlite3_mprintf("SELECT \"%w\", \"%w\" FROM \"%w\";", pk_name, col_name, table_name);
    if (!sql) return SQLITE_NOMEM;
    
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // compute distance function
    vector_distance vd = c->table->options.v_distance;
    vector_type vt = c->table->options.v_type;
    if (vt == VECTOR_TYPE_BIT) vd = VECTOR_DISTANCE_HAMMING;  // Force Hamming for BIT type
    distance_function_t distance_fn = vector_lookup_distance_function(vd, vt);
    if (!distance_fn) {
        rc = sqlite_vtab_set_error(c->base.pVtab, "Distance '%s' is not supported for vector type '%s'", vector_distance_to_name(vd), vector_type_to_name(vt));
        goto cleanup;
    }
    if (vector_use_normalized_cosine(c->table, false)) distance_fn = cosine_normalized_f32;
    int dist_size = (vt == VECTOR_TYPE_BIT) ? ((dimension + 7) / 8) : dimension;

    size_t expected_bytes = vector_bytes_for_dim(vt, dimension);

    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) {rc = SQLITE_OK; goto cleanup;}
        if (rc != SQLITE_ROW) goto cleanup;
        if (sqlite3_column_type(vm, 1) == SQLITE_NULL) continue;

        float *v2 = (float *)sqlite3_column_blob(vm, 1);
        if (v2 == NULL) continue;
        if ((size_t)sqlite3_column_bytes(vm, 1) < expected_bytes) continue;

        float distance = distance_fn((const void *)v1, (const void *)v2, dist_size);
        if (nearly_zero_float32(distance)) distance = 0.0;
        VECTOR_PRINT((void*)v2, vt, dimension);
        
        if (distance < c->distance[0]) vTopKReplaceWorst(c, distance, (int64_t)sqlite3_column_int64(vm, 0));
    }
    
cleanup:
    if (sql) sqlite3_free(sql);
    if (vm) sqlite3_finalize(vm);
    return rc;
}

static int vFullScanCursorFilter (sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    return vCursorFilterCommon(cur, idxNum, idxStr, argc, argv, "vector_full_scan", vFullScanRun, vFullScanSortSlots, vStreamScanCursorRun, false);
}

// MARK: -

static int vQuantRunMemory(vFullScanCursor *c, const preload_index *idx, uint8_t *v, vector_qtype qtype, int dim) {
    const int counter = idx->counter;
    const uint8_t *data = idx->data;
    const size_t rowid_size = sizeof(int64_t);
    const size_t vector_size = (qtype == VECTOR_QUANT_1BIT) ? ((dim + 7) / 8) : (dim * sizeof(uint8_t));
    const size_t total_stride = rowid_size + vector_size;

    // the preloaded index is built from an ordinary writable table, so never trust its
    // row count against its length (the TurboQuant paths already do this)
    const sqlite3_int64 data_bytes = idx->bytes;
    if (!data || counter < 0 || data_bytes < 0 || (sqlite3_uint64)data_bytes < (sqlite3_uint64)counter * (sqlite3_uint64)total_stride) {
        sqlite_vtab_set_error(c->base.pVtab, "Corrupted quantization data preloaded for '%s.%s'", c->table->t_name, c->table->c_name);
        return SQLITE_CORRUPT;
    }

    double current_max = c->distance[0];

    // compute distance function
    vector_distance vd = c->table->options.v_distance;
    vector_type vt = (qtype == VECTOR_QUANT_U8BIT) ? VECTOR_TYPE_U8 : VECTOR_TYPE_I8;
    if (qtype == VECTOR_QUANT_1BIT) {
        vt = VECTOR_TYPE_BIT;
        vd = VECTOR_DISTANCE_HAMMING;
    }
    distance_function_t distance_fn = vector_lookup_distance_function(vd, vt);
    if (!distance_fn) return sqlite_vtab_set_error(c->base.pVtab, "Distance '%s' is not supported for vector type '%s'", vector_distance_to_name(vd), vector_type_to_name(vt));

    for (int i = 0; i < counter; ++i) {
        const uint8_t *current_data = data + (i * total_stride);
        const uint8_t *vector_data = current_data + rowid_size;

        float dist = distance_fn((const void *)v, (const void *)vector_data, (int)vector_size);
        if (nearly_zero_float32(dist)) dist = 0.0;
        
        if (dist < current_max) {
            vTopKReplaceWorst(c, dist, INT64_FROM_INT8PTR(current_data));
            current_max = c->distance[0];
        }
    }

    return SQLITE_OK;
}

static int vTurboPrepareQuery (vFullScanCursor *c, const void *v1, float **qrot_out, float *qnorm_sq_out) {
    int dim = c->table->options.v_dim;
    vector_type type = c->table->options.v_type;
    float *values = (float *)sqlite3_malloc64((sqlite3_uint64)dim * sizeof(float));
    float *rotated = (float *)sqlite3_malloc64((sqlite3_uint64)dim * sizeof(float));
    if (!values || !rotated) {
        if (values) sqlite3_free(values);
        if (rotated) sqlite3_free(rotated);
        return SQLITE_NOMEM;
    }

    float norm_sq = turbo_copy_float(v1, type, dim, values);
    if (c->table->options.v_distance == VECTOR_DISTANCE_COSINE) {
        turbo_normalize_inplace(values, dim, norm_sq);
        norm_sq = 1.0f;
    }
    int rc = table_context_require_turbo_cache(c->table, c->table->options.q_bits, dim);
    if (rc != SQLITE_OK) {
        sqlite3_free(values);
        sqlite3_free(rotated);
        return rc;
    }
    turbo_rotate_with_plan(values, rotated, c->table->turbo_plan);
    sqlite3_free(values);

    *qrot_out = rotated;
    *qnorm_sq_out = norm_sq;
    return SQLITE_OK;
}

static int vTurboRunPackedRows (vFullScanCursor *c, const uint8_t *data, sqlite3_int64 data_bytes, int counter, const float *qrot, float qnorm_sq, const float *centroids, const float *query_lut, const float *norm_lut, int lut_rows, int bits) {
    int dim = c->table->options.v_dim;
    vector_distance distance_type = c->table->options.v_distance;
    size_t packed_bytes = turbo_bytes_for_dim(dim, bits);
    size_t total_stride = sizeof(int64_t) + sizeof(float) + packed_bytes;
    if (!data || counter < 0 || data_bytes < 0 || (sqlite3_uint64)data_bytes < (sqlite3_uint64)counter * (sqlite3_uint64)total_stride) {
        return SQLITE_CORRUPT;
    }

    double current_max = c->distance[0];

    for (int i = 0; i < counter; ++i) {
        const uint8_t *current = data + ((size_t)i * total_stride);
        float scale = 0.0f;
        memcpy(&scale, current + sizeof(int64_t), sizeof(float));
        const uint8_t *packed = current + sizeof(int64_t) + sizeof(float);

        float dist;
        if (query_lut && (distance_type == VECTOR_DISTANCE_DOT || distance_type == VECTOR_DISTANCE_COSINE)) {
            float dot = turbo_dot_from_lut(packed, scale, query_lut, lut_rows, bits, (int)packed_bytes);
            dist = (distance_type == VECTOR_DISTANCE_DOT) ? -dot : (1.0f - dot);
            if (dist < 0.0f && distance_type == VECTOR_DISTANCE_COSINE) dist = 0.0f;
        } else if (query_lut && norm_lut && (distance_type == VECTOR_DISTANCE_L2 || distance_type == VECTOR_DISTANCE_SQUARED_L2)) {
            float dot = turbo_dot_from_lut(packed, scale, query_lut, lut_rows, bits, (int)packed_bytes);
            float norm = turbo_dot_from_lut(packed, 1.0f, norm_lut, lut_rows, bits, (int)packed_bytes);
            double d2 = (double)qnorm_sq + ((double)scale * (double)scale * (double)norm) - 2.0 * (double)dot;
            if (d2 < 0.0) d2 = 0.0;
            dist = (distance_type == VECTOR_DISTANCE_L2) ? (float)sqrt(d2) : (float)d2;
        } else {
            dist = turbo_distance_from_rotated_query(qrot, qnorm_sq, packed, scale, centroids, bits, dim, distance_type);
        }
        if (nearly_zero_float32(dist)) dist = 0.0f;

        if (dist < current_max) {
            vTopKReplaceWorst(c, dist, INT64_FROM_INT8PTR(current));
            current_max = c->distance[0];
        }
    }

    return SQLITE_OK;
}

static int vTurboRun (sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size) {
    (void)v1size;

    int dim = c->table->options.v_dim;
    int bits = c->table->options.q_bits;
    float *qrot = NULL;
    float *query_lut = NULL;
    float *norm_lut = NULL;
    int lut_rows = 0;
    float qnorm_sq = 0.0f;
    sqlite3_stmt *vm = NULL;
    int rc = SQLITE_OK;

    if (bits < 2 || bits > 4) return SQLITE_MISUSE;
    rc = table_context_require_turbo_cache(c->table, bits, dim);
    if (rc != SQLITE_OK) goto cleanup;
    rc = vTurboPrepareQuery(c, v1, &qrot, &qnorm_sq);
    if (rc != SQLITE_OK) goto cleanup;
    if (c->table->options.v_distance == VECTOR_DISTANCE_DOT || c->table->options.v_distance == VECTOR_DISTANCE_COSINE ||
        c->table->options.v_distance == VECTOR_DISTANCE_L2 || c->table->options.v_distance == VECTOR_DISTANCE_SQUARED_L2) {
        query_lut = turbo_build_query_lut(qrot, c->table->turbo_centroids, bits, dim, &lut_rows);
        if (!query_lut) { rc = SQLITE_NOMEM; goto cleanup; }
    }
    if (c->table->options.v_distance == VECTOR_DISTANCE_L2 || c->table->options.v_distance == VECTOR_DISTANCE_SQUARED_L2) {
        int norm_rows = 0;
        norm_lut = turbo_build_norm_lut(c->table->turbo_centroids, bits, dim, &norm_rows);
        if (!norm_lut) { rc = SQLITE_NOMEM; goto cleanup; }
        if (norm_rows != lut_rows) { rc = SQLITE_CORRUPT; goto cleanup; }
    }

    preload_index *idx = preload_index_acquire(c->table);
    if (idx) {
        rc = vTurboRunPackedRows(c, idx->data, idx->bytes, idx->counter, qrot, qnorm_sq, c->table->turbo_centroids, query_lut, norm_lut, lut_rows, bits);
        preload_index_release(idx);
        goto cleanup;
    }

    char sql[STATIC_SQL_SIZE];
    generate_select_quant_table(c->table->t_name, c->table->c_name, sql);
    rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) { rc = SQLITE_OK; break; }
        if (rc != SQLITE_ROW) break;

        int counter = sqlite3_column_int(vm, 0);
        const uint8_t *data = (const uint8_t *)sqlite3_column_blob(vm, 1);
        int bytes = sqlite3_column_bytes(vm, 1);
        if (data) {
            rc = vTurboRunPackedRows(c, data, bytes, counter, qrot, qnorm_sq, c->table->turbo_centroids, query_lut, norm_lut, lut_rows, bits);
            if (rc != SQLITE_OK) break;
        }
    }

cleanup:
    if (rc != SQLITE_OK && c && c->base.pVtab) sqlite_vtab_set_error(c->base.pVtab, "TurboQuant scan failed: %s", sqlite3_errmsg(db));
    if (vm) sqlite3_finalize(vm);
    if (query_lut) sqlite3_free(query_lut);
    if (norm_lut) sqlite3_free(norm_lut);
    if (qrot) sqlite3_free(qrot);
    return rc;
}

static int vQuantRun (sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size) {
    if (c->table->options.q_type == VECTOR_QUANT_TURBO) return vTurboRun(db, c, v1, v1size);

    // quantize target vector
    int dimension = c->table->options.v_dim;
    vector_qtype qtype = c->table->options.q_type;
    size_t alloc_size = (qtype == VECTOR_QUANT_1BIT) ? ((dimension + 7) / 8) : (dimension * sizeof(int8_t));
    uint8_t *v = (uint8_t *)sqlite3_malloc64(alloc_size);
    if (!v) return SQLITE_NOMEM;
    
    // quantize vector
    float offset = c->table->offset;
    float scale = c->table->scale;
    vector_type type = c->table->options.v_type;

    if (qtype == VECTOR_QUANT_1BIT) {
        // 1-bit quantization: convert source to binary based on type
        switch (type) {
            case VECTOR_TYPE_F32: quantize_binary((const float *)v1, v, dimension, c->table->binary_mean); break;
            case VECTOR_TYPE_F16: quantize_binary_f16((const uint16_t *)v1, v, dimension, false); break;
            case VECTOR_TYPE_BF16: quantize_binary_bf16((const uint16_t *)v1, v, dimension, false); break;
            case VECTOR_TYPE_U8: quantize_binary_u8((const uint8_t *)v1, v, dimension); break;
            case VECTOR_TYPE_I8: quantize_binary_i8((const int8_t *)v1, v, dimension); break;
            case VECTOR_TYPE_BIT: memcpy(v, v1, (dimension + 7) / 8); break; // Already binary
        }
    } else {
        // 8-bit quantization (U8BIT or S8BIT)
        switch (type) {
            case VECTOR_TYPE_F32: quantize_float32((const float *)v1, v, offset, scale, dimension, qtype); break;
            case VECTOR_TYPE_F16: quantize_float16((const uint16_t *)v1, v, offset, scale, dimension, qtype); break;
            case VECTOR_TYPE_BF16: quantize_bfloat16((const uint16_t *)v1, v, offset, scale, dimension, qtype); break;
            case VECTOR_TYPE_U8: quantize_u8((const uint8_t *)v1, v, offset, scale, dimension, qtype); break;
            case VECTOR_TYPE_I8: quantize_i8((const int8_t *)v1, v, offset, scale, dimension, qtype); break;
            case VECTOR_TYPE_BIT: memset(v, 0, (size_t)dimension); memcpy(v, v1, (dimension + 7) / 8); break; // see vector_rebuild_quantization
        }
    }

    preload_index *idx = preload_index_acquire(c->table);
    if (idx) {
        int rc = vQuantRunMemory(c, idx, v, qtype, dimension);
        preload_index_release(idx);
        sqlite3_free(v);
        return rc;
    }
    #if DEBUG_VECTOR_SERIALIZATION
    vector_type qprint = (qtype == VECTOR_QUANT_1BIT) ? VECTOR_TYPE_BIT : (qtype == VECTOR_QUANT_U8BIT) ? VECTOR_TYPE_U8 : VECTOR_TYPE_I8;
    VECTOR_PRINT((void*)v, qprint, dimension);
    #endif
    
    char sql[STATIC_SQL_SIZE];
    generate_select_quant_table(c->table->t_name, c->table->c_name, sql);
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto vquant_run_cleanup;
    
    // precompute constants
    const size_t rowid_size = sizeof(int64_t);
    const size_t vector_size = (qtype == VECTOR_QUANT_1BIT) ? ((dimension + 7) / 8) : (dimension * sizeof(uint8_t));
    const size_t total_stride = rowid_size + vector_size;
    
    // compute distance function
    vector_distance vd = c->table->options.v_distance;
    vector_type vt = (qtype == VECTOR_QUANT_U8BIT) ? VECTOR_TYPE_U8 : VECTOR_TYPE_I8;
    if (qtype == VECTOR_QUANT_1BIT) {
        // in case of 1BIT quantization force distance to alway be hamming
        vt = VECTOR_TYPE_BIT;
        vd = VECTOR_DISTANCE_HAMMING;
    }
    distance_function_t distance_fn = vector_lookup_distance_function(vd, vt);
    if (!distance_fn) {
        sqlite_vtab_set_error(c->base.pVtab, "Distance '%s' is not supported for vector type '%s'", vector_distance_to_name(vd), vector_type_to_name(vt));
        sqlite3_finalize(vm);
        sqlite3_free(v);
        return SQLITE_ERROR;
    }
    
    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) {rc = SQLITE_OK; break;} // return error: rebuild must be call (only if first time run)
        else if (rc != SQLITE_ROW) goto vquant_run_cleanup;
        
        int counter = sqlite3_column_int(vm, 0);
        uint8_t *data = (uint8_t *)sqlite3_column_blob(vm, 1);
        int bytes = sqlite3_column_bytes(vm, 1);
        if (!data || counter < 0 || bytes < 0 || (sqlite3_uint64)bytes < (sqlite3_uint64)counter * (sqlite3_uint64)total_stride) {
            sqlite_vtab_set_error(c->base.pVtab, "Corrupted quantization data for '%s.%s'", c->table->t_name, c->table->c_name);
            sqlite3_finalize(vm);
            sqlite3_free(v);
            return SQLITE_CORRUPT;
        }
        
        // cache the maximum value to avoid repeated memory accesses
        double current_max_distance = c->distance[0];
        
        for (int i=0; i<counter; ++i) {
            const uint8_t *current_data = data + (i * total_stride);
            const uint8_t *vector_data = current_data + rowid_size;
            float distance = distance_fn((const void *)v, (const void *)vector_data, (int)vector_size);
            if (nearly_zero_float32(distance)) distance = 0.0;
            VECTOR_PRINT((void*)vector_data, vt, dimension);
            
            if (distance < current_max_distance) {
                vTopKReplaceWorst(c, distance, INT64_FROM_INT8PTR(current_data));
                current_max_distance = c->distance[0]; // update cached max
            }
        }
    }
    
    rc = SQLITE_OK;
    
vquant_run_cleanup:
    if (rc != SQLITE_OK && c && c->base.pVtab) sqlite_vtab_set_error(c->base.pVtab, "Quantized scan failed: %s", sqlite3_errmsg(db));
    if (vm) sqlite3_finalize(vm);
    if (v) sqlite3_free(v);
    return rc;
}


static int vQuantCursorFilter (sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    return vCursorFilterCommon(cur, idxNum, idxStr, argc, argv, "vector_quantize_scan", vQuantRun, vFullScanSortSlots, vStreamQuantCursorRun, true);
}

static int vStreamScanCursorRun (sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size) {
    // duplicate input vector (to be later used by the Next callback)
    void *v = sqlite_memdup(v1, v1size);
    if (!v) return SQLITE_NOMEM;
    
    const char *pk_name = c->table->pk_name;
    const char *col_name = c->table->c_name;
    const char *table_name = c->table->t_name;
    int dimension = c->table->options.v_dim;
    
    c->stream.vector = (void *)v;
    c->stream.vsize = v1size;
    c->stream.vdim = dimension;
    
    char *sql = sqlite3_mprintf("SELECT \"%w\", \"%w\" FROM \"%w\";", pk_name, col_name, table_name);
    if (!sql) {
        sqlite3_free(v);
        c->stream.vector = NULL;
        return SQLITE_NOMEM;
    }
    
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // compute distance function
    vector_distance vd = c->table->options.v_distance;
    vector_type vt = c->table->options.v_type;
    if (vt == VECTOR_TYPE_BIT) vd = VECTOR_DISTANCE_HAMMING;  // Force Hamming for BIT type
    distance_function_t distance_fn = vector_lookup_distance_function(vd, vt);
    if (!distance_fn) {
        rc = sqlite_vtab_set_error(c->base.pVtab, "Distance '%s' is not supported for vector type '%s'", vector_distance_to_name(vd), vector_type_to_name(vt));
        goto cleanup;
    }
    if (vector_use_normalized_cosine(c->table, false)) distance_fn = cosine_normalized_f32;

    c->stream.distance_fn = distance_fn;
    c->stream.vm = vm;

    if (sql) sqlite3_free(sql);
    return SQLITE_OK;

cleanup:
    if (sql) sqlite3_free(sql);
    if (vm) sqlite3_finalize(vm);
    if (v) sqlite3_free(v);
    c->stream.vector = NULL;
    return rc;
}

static int vStreamTurboCursorRun (sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size) {
    (void)v1size;

    float *qrot = NULL;
    float qnorm_sq = 0.0f;
    int dim = c->table->options.v_dim;
    int bits = c->table->options.q_bits;
    int rc = table_context_require_turbo_cache(c->table, bits, dim);
    if (rc != SQLITE_OK) return rc;

    rc = vTurboPrepareQuery(c, v1, &qrot, &qnorm_sq);
    if (rc != SQLITE_OK) return rc;

    c->stream.vector = qrot;
    c->stream.vsize = (int)turbo_bytes_for_dim(dim, bits);
    c->stream.vdim = dim;
    c->stream.turbo_qnorm_sq = qnorm_sq;
    c->stream.turbo_bits = bits;
    if (c->table->options.v_distance == VECTOR_DISTANCE_DOT || c->table->options.v_distance == VECTOR_DISTANCE_COSINE ||
        c->table->options.v_distance == VECTOR_DISTANCE_L2 || c->table->options.v_distance == VECTOR_DISTANCE_SQUARED_L2) {
        c->stream.turbo_query_lut = turbo_build_query_lut(qrot, c->table->turbo_centroids, bits, dim, &c->stream.turbo_lut_rows);
        if (!c->stream.turbo_query_lut) {
            sqlite3_free(qrot);
            c->stream.vector = NULL;
            return SQLITE_NOMEM;
        }
    }
    if (c->table->options.v_distance == VECTOR_DISTANCE_L2 || c->table->options.v_distance == VECTOR_DISTANCE_SQUARED_L2) {
        int norm_rows = 0;
        int norm_rc = SQLITE_OK;
        c->stream.turbo_norm_lut = turbo_build_norm_lut(c->table->turbo_centroids, bits, dim, &norm_rows);
        if (!c->stream.turbo_norm_lut) norm_rc = SQLITE_NOMEM;
        else if (norm_rows != c->stream.turbo_lut_rows) norm_rc = SQLITE_CORRUPT;
        if (norm_rc != SQLITE_OK) {
            sqlite3_free(qrot);
            if (c->stream.turbo_query_lut) {
                sqlite3_free(c->stream.turbo_query_lut);
                c->stream.turbo_query_lut = NULL;
            }
            if (c->stream.turbo_norm_lut) {
                sqlite3_free(c->stream.turbo_norm_lut);
                c->stream.turbo_norm_lut = NULL;
            }
            c->stream.turbo_lut_rows = 0;
            c->stream.vector = NULL;
            return norm_rc;
        }
    }

    preload_index *idx = preload_index_acquire(c->table);
    if (idx) {
        c->stream.preload_ref = idx;
        c->stream.dindex = 0;
        c->stream.data = idx->data;
        c->stream.dcounter = idx->counter;
        c->stream.data_bytes = idx->bytes;
        return SQLITE_OK;
    }

    char sql[STATIC_SQL_SIZE];
    generate_select_quant_table(c->table->t_name, c->table->c_name, sql);
    sqlite3_stmt *vm = NULL;
    rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) {
        sqlite3_free(qrot);
        if (c->stream.turbo_query_lut) {
            sqlite3_free(c->stream.turbo_query_lut);
            c->stream.turbo_query_lut = NULL;
            c->stream.turbo_lut_rows = 0;
        }
        if (c->stream.turbo_norm_lut) {
            sqlite3_free(c->stream.turbo_norm_lut);
            c->stream.turbo_norm_lut = NULL;
        }
        c->stream.vector = NULL;
        if (vm) sqlite3_finalize(vm);
        return rc;
    }

    c->stream.vm = vm;
    return SQLITE_OK;
}

static int vStreamQuantCursorRun (sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size) {
    if (c->table->options.q_type == VECTOR_QUANT_TURBO) return vStreamTurboCursorRun(db, c, v1, v1size);

    // quantize input vector
    int dimension = c->table->options.v_dim;
    vector_qtype qtype = c->table->options.q_type;
    size_t alloc_size = (qtype == VECTOR_QUANT_1BIT) ? ((dimension + 7) / 8) : (dimension * sizeof(int8_t));
    uint8_t *v = (uint8_t *)sqlite3_malloc64(alloc_size);
    if (!v) return SQLITE_NOMEM;
    
    // quantize vector
    float offset = c->table->offset;
    float scale = c->table->scale;
    vector_type type = c->table->options.v_type;

    if (qtype == VECTOR_QUANT_1BIT) {
        // 1-bit quantization: convert source to binary based on type
        switch (type) {
            case VECTOR_TYPE_F32: quantize_binary((const float *)v1, v, dimension, false); break;
            case VECTOR_TYPE_F16: quantize_binary_f16((const uint16_t *)v1, v, dimension, false); break;
            case VECTOR_TYPE_BF16: quantize_binary_bf16((const uint16_t *)v1, v, dimension, false); break;
            case VECTOR_TYPE_U8: quantize_binary_u8((const uint8_t *)v1, v, dimension); break;
            case VECTOR_TYPE_I8: quantize_binary_i8((const int8_t *)v1, v, dimension); break;
            case VECTOR_TYPE_BIT: memcpy(v, v1, (dimension + 7) / 8); break; // Already binary
        }
    } else {
        // 8-bit quantization (U8BIT or S8BIT)
        switch (type) {
            case VECTOR_TYPE_F32: quantize_float32((const float *)v1, v, offset, scale, dimension, qtype); break;
            case VECTOR_TYPE_F16: quantize_float16((const uint16_t *)v1, v, offset, scale, dimension, qtype); break;
            case VECTOR_TYPE_BF16: quantize_bfloat16((const uint16_t *)v1, v, offset, scale, dimension, qtype); break;
            case VECTOR_TYPE_U8: quantize_u8((const uint8_t *)v1, v, offset, scale, dimension, qtype); break;
            case VECTOR_TYPE_I8: quantize_i8((const int8_t *)v1, v, offset, scale, dimension, qtype); break;
            case VECTOR_TYPE_BIT: memset(v, 0, (size_t)dimension); memcpy(v, v1, (dimension + 7) / 8); break; // see vector_rebuild_quantization
        }
    }

    c->stream.vector = (void *)v;
    c->stream.vsize = (qtype == VECTOR_QUANT_1BIT) ? (int)((dimension + 7) / 8) : (int)(dimension * sizeof(int8_t));
    c->stream.vdim = dimension;
    
    // compute distance function
    vector_distance vd = c->table->options.v_distance;
    vector_type vt = (qtype == VECTOR_QUANT_U8BIT) ? VECTOR_TYPE_U8 : VECTOR_TYPE_I8;
    if (qtype == VECTOR_QUANT_1BIT) {
        // in case of 1BIT quantization force distance to always be hamming
        vt = VECTOR_TYPE_BIT;
        vd = VECTOR_DISTANCE_HAMMING;
    }
    distance_function_t distance_fn = vector_lookup_distance_function(vd, vt);
    if (!distance_fn) {
        sqlite3_free(v);
        c->stream.vector = NULL;
        return sqlite_vtab_set_error(c->base.pVtab, "Distance '%s' is not supported for vector type '%s'", vector_distance_to_name(vd), vector_type_to_name(vt));
    }
    c->stream.distance_fn = distance_fn;
    
    // check if quant representation was preloaded
    preload_index *idx = preload_index_acquire(c->table);
    if (idx) {
        c->stream.preload_ref = idx;
        c->stream.dindex = 0;
        c->stream.data = idx->data;
        c->stream.dcounter = idx->counter;
        c->stream.data_bytes = idx->bytes;
        return SQLITE_OK;
    }
    
    char sql[STATIC_SQL_SIZE];
    generate_select_quant_table(c->table->t_name, c->table->c_name, sql);
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    c->stream.vm = vm;
    return SQLITE_OK;
    
cleanup:
    if (vm) sqlite3_finalize(vm);
    if (v) sqlite3_free(v);
    c->stream.vector = NULL;
    return rc;
}

// ---------------------------

static sqlite3_module vFullScanModule = {
  /* iVersion    */ 0,
  /* xCreate     */ 0,
  /* xConnect    */ vFullScanConnect,
  /* xBestIndex  */ vFullScanBestIndex,
  /* xDisconnect */ vFullScanDisconnect,
  /* xDestroy    */ 0,
  /* xOpen       */ vFullScanCursorOpen,
  /* xClose      */ vFullScanCursorClose,
  /* xFilter     */ vFullScanCursorFilter,
  /* xNext       */ vFullScanCursorNext,
  /* xEof        */ vFullScanCursorEof,
  /* xColumn     */ vFullScanCursorColumn,
  /* xRowid      */ vFullScanCursorRowid,
  /* xUpdate     */ 0,
  /* xBegin      */ 0,
  /* xSync       */ 0,
  /* xCommit     */ 0,
  /* xRollback   */ 0,
  /* xFindMethod */ 0,
  /* xRename     */ 0,
  /* xSavepoint  */ 0,
  /* xRelease    */ 0,
  /* xRollbackTo */ 0,
  /* xShadowName */ 0,
  /* xIntegrity  */ 0
};

static sqlite3_module vQuantScanModule = {
  /* iVersion    */ 0,
  /* xCreate     */ 0,
  /* xConnect    */ vFullScanConnect,
  /* xBestIndex  */ vFullScanBestIndex,
  /* xDisconnect */ vFullScanDisconnect,
  /* xDestroy    */ 0,
  /* xOpen       */ vFullScanCursorOpen,
  /* xClose      */ vFullScanCursorClose,
  /* xFilter     */ vQuantCursorFilter,
  /* xNext       */ vFullScanCursorNext,
  /* xEof        */ vFullScanCursorEof,
  /* xColumn     */ vFullScanCursorColumn,
  /* xRowid      */ vFullScanCursorRowid,
  /* xUpdate     */ 0,
  /* xBegin      */ 0,
  /* xSync       */ 0,
  /* xCommit     */ 0,
  /* xRollback   */ 0,
  /* xFindMethod */ 0,
  /* xRename     */ 0,
  /* xSavepoint  */ 0,
  /* xRelease    */ 0,
  /* xRollbackTo */ 0,
  /* xShadowName */ 0,
  /* xIntegrity  */ 0
};

// MARK: -

static void vector_init (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT, SQLITE_TEXT};
    if (sanity_check_args(context, "vector_init", argc, argv, 3, types) == false) return;
    
    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);
    const char *arg_options = (const char *)sqlite3_value_text(argv[2]);
    
    // sanity check table and column
    if (sqlite_sanity_check(context, table_name, column_name) == false) return;
    
    // parse options
    vector_options options = vector_options_create();
    bool rc = parse_keyvalue_string(context, arg_options, vector_keyvalue_callback, &options);
    if (rc == false) return;
    
    // sanity check mandatory fields
    if (options.v_type == 0) {
        context_result_error(context, SQLITE_ERROR, "Vector type value is mandatory in vector_init");
        return;
    }
    
    if (options.v_dim == 0) {
        context_result_error(context, SQLITE_ERROR, "Vector dimension value is mandatory in vector_init");
        return;
    }
    
    if (vector_distance_is_supported(options.v_distance, options.v_type) == false) {
        context_result_error(context, SQLITE_ERROR, "Distance '%s' is not supported for vector type '%s'", vector_distance_to_name(options.v_distance), vector_type_to_name(options.v_type));
        return;
    }
    
    // check if table is already loaded
    vector_context *v_ctx = (vector_context *)sqlite3_user_data(context);
    table_context *t_ctx = vector_context_lookup(v_ctx, table_name, column_name);
    if (t_ctx) {
        // sanity check
        if (options.v_dim != t_ctx->options.v_dim) {
            context_result_error(context, SQLITE_ERROR, "Inconsistent vector dimension for '%s.%s': existing=%d, provided=%d", table_name, column_name, t_ctx->options.v_dim, options.v_dim);
            return;
        }
        
        if (options.v_type != t_ctx->options.v_type) {
            context_result_error(context, SQLITE_ERROR, "Inconsistent vector type for '%s.%s': existing=%s, provided=%s", table_name, column_name, vector_type_to_name(t_ctx->options.v_type), vector_type_to_name(options.v_type));
            return;
        }
        
        if (options.v_normalized != t_ctx->options.v_normalized) {
            context_result_error(context, SQLITE_ERROR, "Inconsistent normalization flag for '%s.%s': existing=%s, provided=%s", table_name, column_name, t_ctx->options.v_normalized ? "true" : "false", options.v_normalized ? "true" : "false");
            return;
        }
        
        // no need to add a new entry
        return;
    }
    
    vector_context_add(context, v_ctx, table_name, column_name, &options);
}

static void vector_version (sqlite3_context *context, int argc, sqlite3_value **argv) {
    sqlite3_result_text(context, SQLITE_VECTOR_VERSION, -1, NULL);
}

static void vector_backend (sqlite3_context *context, int argc, sqlite3_value **argv) {
    sqlite3_result_text(context, distance_backend_name, -1, NULL);
}

static void vector_turboquant_backend (sqlite3_context *context, int argc, sqlite3_value **argv) {
    sqlite3_result_text(context, turbo_lut_backend_name, -1, NULL);
}
    
// MARK: -

SQLITE_VECTOR_API int sqlite3_vector_init (sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi) {
    #ifndef SQLITE_CORE
    SQLITE_EXTENSION_INIT2(pApi);
    #endif
    int rc = SQLITE_OK;
    
    // there's no built-in way to verify if sqlite3_vector_init has already been called for this specific database connection
    // the workaround is to attempt to execute vector_version and check for an error
    // an error indicates that initialization has not been performed
    if (sqlite3_exec(db, "SELECT vector_version();", NULL, NULL, NULL) == SQLITE_OK) return SQLITE_OK;
    
    // get an app global static mutex
    qmutex = sqlite3_mutex_alloc(SQLITE_MUTEX_STATIC_APP1);
    
    // init internal distance functions (do not force CPU)
    init_distance_functions(false);
    
    // create internal table
    rc = sqlite3_exec(db, VECTOR_INTERNAL_TABLE, NULL, NULL, NULL);
    if (rc != SQLITE_OK) return rc;
    
    // init context
    void *ctx = vector_context_create();
    if (!ctx) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("Out of memory: failed to allocate vector extension context.");
        return SQLITE_NOMEM;
    }
    
    rc = sqlite3_create_function_v2(db, "vector_version", 0, SQLITE_UTF8, ctx, vector_version, NULL, NULL, vector_context_free);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "vector_backend", 0, SQLITE_UTF8, ctx, vector_backend, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "vector_turboquant_backend", 0, SQLITE_UTF8, ctx, vector_turboquant_backend, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // table_name, column_name, options
    rc = sqlite3_create_function(db, "vector_init", 3, SQLITE_UTF8, ctx, vector_init, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // table_name, column_name, options
    rc = sqlite3_create_function(db, "vector_quantize", 3, SQLITE_UTF8, ctx, vector_quantize3, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // table_name, column_name
    rc = sqlite3_create_function(db, "vector_quantize", 2, SQLITE_UTF8, ctx, vector_quantize2, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // table_name, column_name
    rc = sqlite3_create_function(db, "vector_quantize_memory", 2, SQLITE_UTF8, ctx, vector_quantize_memory, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // table_name, column_name
    rc = sqlite3_create_function(db, "vector_quantize_preload", 2, SQLITE_UTF8, ctx, vector_quantize_preload, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // table_name, column_name
    rc = sqlite3_create_function(db, "vector_quantize_cleanup", 2, SQLITE_UTF8, ctx, vector_quantize_cleanup, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "vector_as_f32", 1, SQLITE_UTF8, ctx, vector_as_f32, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    rc = sqlite3_create_function(db, "vector_as_f32", 2, SQLITE_UTF8, ctx, vector_as_f32, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "vector_as_f16", 1, SQLITE_UTF8, ctx, vector_as_f16, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    rc = sqlite3_create_function(db, "vector_as_f16", 2, SQLITE_UTF8, ctx, vector_as_f16, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "vector_as_bf16", 1, SQLITE_UTF8, ctx, vector_as_bf16, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    rc = sqlite3_create_function(db, "vector_as_bf16", 2, SQLITE_UTF8, ctx, vector_as_bf16, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "vector_as_i8", 1, SQLITE_UTF8, ctx, vector_as_i8, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    rc = sqlite3_create_function(db, "vector_as_i8", 2, SQLITE_UTF8, ctx, vector_as_i8, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "vector_as_u8", 1, SQLITE_UTF8, ctx, vector_as_u8, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    rc = sqlite3_create_function(db, "vector_as_u8", 2, SQLITE_UTF8, ctx, vector_as_u8, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_function(db, "vector_as_bit", 1, SQLITE_UTF8, ctx, vector_as_bit, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    rc = sqlite3_create_function(db, "vector_as_bit", 2, SQLITE_UTF8, ctx, vector_as_bit, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;

    rc = sqlite3_create_module(db, "vector_full_scan", &vFullScanModule, ctx);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_module(db, "vector_quantize_scan", &vQuantScanModule, ctx);
    if (rc != SQLITE_OK) goto cleanup;

    // backward-compat aliases: _stream modules merged into main modules in 0.9.80
    rc = sqlite3_create_module(db, "vector_full_scan_stream", &vFullScanModule, ctx);
    if (rc != SQLITE_OK) goto cleanup;
    rc = sqlite3_create_module(db, "vector_quantize_scan_stream", &vQuantScanModule, ctx);
    if (rc != SQLITE_OK) goto cleanup;

    return SQLITE_OK;

cleanup:
    // do NOT free ctx here: it is owned by the destructor registered with
    // sqlite3_create_function_v2() above, which SQLite invokes both when that call itself
    // fails and when the connection is closed. Freeing it again would be a double free.
    return rc;
}

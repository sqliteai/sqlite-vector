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
#include <stdio.h>
#include <ctype.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <stdint.h>
#include <stdbool.h>

#define DEBUG_VECTOR_ALWAYS(...)                    do {printf(__VA_ARGS__ );printf("\n");} while (0)

#if ENABLE_VECTOR_DEBUG
#define DEBUG_VECTOR(...)                           do {printf(__VA_ARGS__ );printf("\n");} while (0)
#else
#define DEBUG_VECTOR(...)
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
#define OPTION_KEY_QUANTTYPE                        "quant_type"
#define OPTION_KEY_DISTANCE                         "distance"

typedef struct {
    vector_type     v_type;                 // vector type
    int             v_dim;                  // vector dimension
    bool            v_normalized;           // is vector normalized ?
    vector_distance v_distance;             // vector distance function
    
    vector_qtype    q_type;                 // quantization type
    uint64_t        max_memory;             // max memory
} vector_options;

typedef struct {
    char            *t_name;                // table name
    char            *c_name;                // column name
    char            *pk_name;               // INTEGER primary key name (in case of WITHOUT ROWID tables) or rowid if NULL
    
    vector_options  options;                // options parsed in key=value arguments
    float           scale;                  // computed value by quantization
    float           offset;                 // computed value by quantization
    
    void            *preloaded;
    int             precounter;
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
    
    sqlite3_int64       *rowids;
    double              *distance;
    
    int                 size;
    int                 max_index;
    int                 row_index;
    int                 row_count;
    
    table_context       *table;
} vFullScanCursor;

typedef bool (*keyvalue_callback)(sqlite3_context *context, void *xdata, const char *key, int key_len, const char *value, int value_len);
typedef int (*vcursor_run_callback)(sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size);
typedef int (*vcursor_sort_callback)(vFullScanCursor *c);

extern distance_function_t dispatch_distance_table[VECTOR_DISTANCE_MAX][VECTOR_TYPE_MAX];
extern char *distance_backend_name;

// MARK: - FLOAT / BFLOAT16 -

// there is a native bfloat16_t ARM intrinsic
typedef uint16_t float16_t;
typedef uint16_t bfloat16_t;

static inline bfloat16_t float32_to_bfloat16 (float f) {
    uint32_t bits = *(uint32_t *)&f;
    return (bfloat16_t)(bits >> 16);
}

static inline float bfloat16_to_float32 (bfloat16_t bf) {
    uint32_t bits = ((uint32_t)bf) << 16;
    return *(float *)&bits;
}

// convert 16-bit float (IEEE-754 binary16) to 32-bit float
static inline float float16_to_float32 (uint16_t h) {
    uint16_t h_exp = (h & 0x7C00u);
    uint16_t h_sig = (h & 0x03FFu);
    uint32_t f_sgn = ((uint32_t)h & 0x8000u) << 16;

    if (h_exp == 0x0000u) {
        // Zero or subnormal
        if (h_sig == 0) {
            return *((float*)&f_sgn);  // ±0.0
        } else {
            // Normalize subnormal
            float result = (float)(h_sig) / 1024.0f;
            return *((float*)&f_sgn) ? -result : result;
        }
    } else if (h_exp == 0x7C00u) {
        // Inf or NaN
        uint32_t f_inf_nan = f_sgn | 0x7F800000u | ((uint32_t)(h_sig) << 13);
        return *((float*)&f_inf_nan);
    }

    // Normalized number
    uint32_t f_exp = ((uint32_t)(h_exp >> 10) + (127 - 15)) << 23;
    uint32_t f_sig = ((uint32_t)h_sig) << 13;
    uint32_t f = f_sgn | f_exp | f_sig;

    float result;
    *((uint32_t*)&result) = f;
    return result;
}

// convert 32-bit float to 16-bit float (IEEE-754 binary16), result stored as uint16_t
static inline uint16_t float32_to_float16(float f) {
    union {
        uint32_t u;
        float f;
    } v;
    v.f = f;

    uint32_t f_bits = v.u;

    uint32_t sign = (f_bits >> 16) & 0x8000;
    int32_t  exp  = ((f_bits >> 23) & 0xFF) - 127 + 15;
    uint32_t frac = f_bits & 0x007FFFFF;

    if (exp <= 0) {
        // Subnormal or underflow
        if (exp < -10) {
            // Too small for subnormal — flush to zero
            return (uint16_t)sign;
        }
        // Subnormal
        frac |= 0x00800000; // add implicit leading 1
        int shift = 14 - exp;
        uint16_t subnormal = (uint16_t)(frac >> shift);
        // Round to nearest
        if ((frac >> (shift - 1)) & 1) subnormal += 1;
        return sign | subnormal;
    } else if (exp >= 31) {
        // Overflow → Inf or NaN
        if (frac == 0) {
            return (uint16_t)(sign | 0x7C00); // Inf
        } else {
            return (uint16_t)(sign | 0x7C00 | (frac >> 13)); // NaN
        }
    }

    // Normal range: round and pack
    uint16_t h_exp = (uint16_t)(exp << 10);
    uint16_t h_frac = (uint16_t)(frac >> 13);
    // Round to nearest
    if (frac & 0x00001000) {
        h_frac += 1;
        if (h_frac == 0x0400) {  // mantissa overflow
            h_frac = 0;
            h_exp += 0x0400;
            if (h_exp >= 0x7C00) h_exp = 0x7C00; // clamp to Inf
        }
    }

    return (uint16_t)(sign | h_exp | h_frac);
}

// MARK: - Quantization -

static inline void quantize_float32_to_u8 (float *v, uint8_t *q, float offset, float scale, int n) {
    int i = 0;
    for (; i + 3 < n; i += 4) {
        float s0 = (v[i]     - offset) * scale;
        float s1 = (v[i + 1] - offset) * scale;
        float s2 = (v[i + 2] - offset) * scale;
        float s3 = (v[i + 3] - offset) * scale;

        int r0 = (int)(s0 + 0.5f * (1.0f - 2.0f * (s0 < 0.0f)));
        int r1 = (int)(s1 + 0.5f * (1.0f - 2.0f * (s1 < 0.0f)));
        int r2 = (int)(s2 + 0.5f * (1.0f - 2.0f * (s2 < 0.0f)));
        int r3 = (int)(s3 + 0.5f * (1.0f - 2.0f * (s3 < 0.0f)));

        r0 = r0 > 255 ? 255 : (r0 < 0 ? 0 : r0);
        r1 = r1 > 255 ? 255 : (r1 < 0 ? 0 : r1);
        r2 = r2 > 255 ? 255 : (r2 < 0 ? 0 : r2);
        r3 = r3 > 255 ? 255 : (r3 < 0 ? 0 : r3);

        q[i]     = (uint8_t)r0;
        q[i + 1] = (uint8_t)r1;
        q[i + 2] = (uint8_t)r2;
        q[i + 3] = (uint8_t)r3;
    }

    // Handle remaining elements
    for (; i < n; ++i) {
        float scaled = (v[i] - offset) * scale;
        int rounded = (int)(scaled + 0.5f * (1.0f - 2.0f * (scaled < 0.0f)));
        rounded = rounded > 255 ? 255 : (rounded < 0 ? 0 : rounded);
        q[i] = (uint8_t)rounded;
    }
}

static inline void quantize_float16_to_u8 (const int16_t *v, uint8_t *q, float offset, float scale, int n) {
    for (int i = 0; i < n; ++i) {
        float x = (float16_to_float32((uint16_t)v[i]) - offset) * scale;
        int r = (int)(x + 0.5f * (1.0f - 2.0f * (x < 0.0f)));
        q[i] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
    }
}

static inline void quantize_bfloat16_to_u8(const int16_t *v, uint8_t *q, float offset, float scale, int n) {
    for (int i = 0; i < n; ++i) {
        float x = (bfloat16_to_float32(v[i]) - offset) * scale;
        int r = (int)(x + 0.5f * (1.0f - 2.0f * (x < 0.0f)));  // round to nearest
        q[i] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));     // clamp to [0, 255]
    }
}

static inline void quantize_u8_to_u8 (const uint8_t *v, uint8_t *q, float offset, float scale, int n) {
    for (int i = 0; i < n; ++i) {
        float x = ((float)v[i] - offset) * scale;
        int r = (int)(x + 0.5f * (1.0f - 2.0f * (x < 0.0f)));
        q[i] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
    }
}

static inline void quantize_i8_to_u8 (const int8_t *v, uint8_t *q, float offset, float scale, int n) {
    for (int i = 0; i < n; ++i) {
        float x = ((float)v[i] - offset) * scale;
        int r = (int)(x + 0.5f * (1.0f - 2.0f * (x < 0.0f)));
        q[i] = (uint8_t)(r < 0 ? 0 : (r > 255 ? 255 : r));
    }
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
    sqlite3_snprintf(sizeof(sql), sql, "SELECT COUNT(*), type, name FROM pragma_table_info('%q') WHERE pk > 0;", table_name);
    char *prikey = NULL;
    
    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) == SQLITE_OK) {
        sqlite3_bind_text(stmt, 1, table_name, -1, SQLITE_STATIC);
        
        if (sqlite3_step(stmt) == SQLITE_ROW) {
            int count = sqlite3_column_int(stmt, 0);
            if (count == 1) {
                const char *decl_type = (const char *)sqlite3_column_text(stmt, 1);
                // see https://www.sqlite.org/datatype3.html (Determination Of Column Affinity)
                if (strcasestr(decl_type, "INT")) {
                    prikey = sqlite_strdup((const char *)sqlite3_column_text(stmt, 2));
                }
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
        context_result_error(context, SQLITE_ERROR, "Table '%s' does not exist.", table_name);
        return false;
    }
    
    // column_name must exists
    if (sqlite_column_exists(db, table_name, column_name) == false) {
        context_result_error(context, SQLITE_ERROR, "Column '%s' does not exist in table '%s'.", column_name, table_name);
        return false;
    }
    
    // column_name must be of type BLOB
    if (sqlite_column_is_blob(db, table_name, column_name) == false) {
        context_result_error(context, SQLITE_ERROR, "Column '%s' in table '%s' must be of type BLOB.", column_name, table_name);
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

// MARK: - General Utils -

static size_t vector_type_to_size (vector_type type) {
    switch (type) {
        case VECTOR_TYPE_F32: return sizeof(float);
        case VECTOR_TYPE_F16: return sizeof(uint16_t);
        case VECTOR_TYPE_BF16: return sizeof(uint16_t/*bfloat16*/); // TODO: FIX ME
        case VECTOR_TYPE_U8: return sizeof(uint8_t);
        case VECTOR_TYPE_I8: return sizeof(int8_t);
    }
    return 0;
}

static vector_type vector_name_to_type (const char *vname) {
    if (strcasecmp(vname, "FLOAT32") == 0) return VECTOR_TYPE_F32;
    if (strcasecmp(vname, "FLOAT16") == 0) return VECTOR_TYPE_F16;
    if (strcasecmp(vname, "FLOATB16") == 0) return VECTOR_TYPE_BF16;
    if (strcasecmp(vname, "UINT8") == 0) return VECTOR_TYPE_U8;
    if (strcasecmp(vname, "INT8") == 0) return VECTOR_TYPE_I8;
    return 0;
}

const char *vector_type_to_name (vector_type type) {
    switch (type) {
        case VECTOR_TYPE_F32: return "FLOAT32";
        case VECTOR_TYPE_F16: return "FLOAT16";
        case VECTOR_TYPE_BF16: return "FLOATB16";
        case VECTOR_TYPE_U8: return "UINT8";
        case VECTOR_TYPE_I8: return "INT8";
    }
    return "N/A";
}

static vector_qtype quant_name_to_type (const char *qname) {
    if (strcasecmp(qname, "QUANTU8") == 0) return VECTOR_QUANT_8BIT;
    return 0;
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
    return 0;
}

const char *vector_distance_to_name (vector_distance type) {
    switch (type) {
        case VECTOR_DISTANCE_L2: return "L2";
        case VECTOR_DISTANCE_SQUARED_L2: return "L2 SQUARED";
        case VECTOR_DISTANCE_COSINE: return "COSINE";
        case VECTOR_DISTANCE_DOT: return "DOT";
        case VECTOR_DISTANCE_L1: return "L1";
    }
    return "N/A";
}

static bool sanity_check_args (sqlite3_context *context, const char *func_name, int argc, sqlite3_value **argv, int ntypes, int *types) {
    if (argc != ntypes) {
        context_result_error(context, SQLITE_ERROR, "Function '%s' expects %d arguments, but %d were provided.", func_name, ntypes, argc);
        return false;
    }
    
    for (int i=0; i<argc; ++i) {
        int actual_type = sqlite3_value_type(argv[i]);
        if (actual_type != types[i]) {
            context_result_error(context, SQLITE_ERROR, "Function '%s': argument %d must be of type %s (got %s).", func_name, (i+1), sqlite_type_name(types[i]), sqlite_type_name(actual_type));
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
    size_t len = (value_len > sizeof(buffer)-1) ? sizeof(buffer)-1 : value_len;
    memcpy(buffer, value, len);
    
    if (strncasecmp(key, OPTION_KEY_TYPE, key_len) == 0) {
        vector_type type = vector_name_to_type(buffer);
        if (type == 0) return context_result_error(context, SQLITE_ERROR, "Invalid vector type: '%s' is not a recognized type.", buffer);
        options->v_type = type;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_DIMENSION, key_len) == 0) {
        int dimension = (int)strtol(buffer, NULL, 0);
        if (dimension <= 0) return context_result_error(context, SQLITE_ERROR, "Invalid vector dimension: expected a positive integer, got '%s'.", buffer);
        options->v_dim = dimension;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_NORMALIZED, key_len) == 0) {
        int normalized = (int)strtol(buffer, NULL, 0);
        options->v_normalized = (normalized != 0);
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_MAXMEMORY, key_len) == 0) {
        uint64_t max_memory = human_to_number(buffer);
        if (max_memory >= 0) options->max_memory = (int)max_memory;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_QUANTTYPE, key_len) == 0) {
        vector_qtype type = quant_name_to_type(buffer);
        if (type == 0) return context_result_error(context, SQLITE_ERROR, "Invalid quantization type: '%s' is not a recognized or supported quantization type.", buffer);
        options->q_type = type;
        return true;
    }
    
    if (strncasecmp(key, OPTION_KEY_DISTANCE, key_len) == 0) {
        vector_distance type = distance_name_to_type(buffer);
        if (type == 0) return context_result_error(context, SQLITE_ERROR, "Invalid distance name: '%s' is not a recognized or supported distance.", buffer);
        options->v_distance = type;
        return true;
    }
    
    // means ignore unknown keys
    return true;
}

// MARK: - SQL -

static char *generate_create_quant_table (const char *table_name, const char *column_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "CREATE TABLE IF NOT EXISTS vector0_%q_%q (rowid1 INTEGER, rowid2 INTEGER, counter INTEGER, data BLOB);", table_name, column_name);
}

static char *generate_drop_quant_table (const char *table_name, const char *column_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "DROP TABLE IF EXISTS vector0_%q_%q;", table_name, column_name);
}

static char *generate_select_from_table (const char *table_name, const char *column_name, const char *pk_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "SELECT %q, %q FROM %q ORDER BY %q;", pk_name, column_name, table_name, pk_name);
}

static char *generate_select_quant_table (const char *table_name, const char *column_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "SELECT counter, data FROM vector0_%q_%q;", table_name, column_name);
}

static char *generate_memory_quant_table (const char *table_name, const char *column_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "SELECT SUM(LENGTH(data)) FROM vector0_%q_%q;", table_name, column_name);
}

static char * generate_insert_quant_table (const char *table_name, const char *column_name, char sql[STATIC_SQL_SIZE]) {
    return sqlite3_snprintf(STATIC_SQL_SIZE, sql, "INSERT INTO vector0_%q_%q (rowid1, rowid2, counter, data) VALUES (?, ?, ?, ?);", table_name, column_name);
}

// MARK: - Vector Context and Options -

void *vector_context_create (void) {
    vector_context *ctx = (vector_context *)sqlite3_malloc(sizeof(vector_context));
    if (!ctx) return NULL;
    
    memset(ctx, 0, sizeof(vector_context));
    return (void *)ctx;
}

table_context *vector_context_lookup (vector_context *ctx, const char *table_name, const char *column_name) {
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
        context_result_error(context, SQLITE_ERROR, "Cannot add table: maximum number of allowed tables reached (%d).", MAX_TABLES);
        return;
    }
    
    char *t_name = sqlite_strdup(table_name);
    char *c_name = sqlite_strdup(column_name);
    if (!t_name || !c_name) {
        context_result_error(context, SQLITE_NOMEM, "Out of memory: unable to duplicate table or column name.");
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
        (is_without_rowid) ? context_result_error(context, SQLITE_NOMEM, "Out of memory: unable to duplicate rowid column name.") : context_result_error(context, SQLITE_ERROR, "WITHOUT ROWID table '%s' must have exactly one PRIMARY KEY column of type INTEGER.", table_name);
        return;
    }
    
    int index = ctx->table_count;
    ctx->tables[index].t_name = t_name;
    ctx->tables[index].c_name = c_name;
    ctx->tables[index].pk_name = prikey;
    ctx->tables[index].options = *options;
    ctx->table_count++;
}

void vector_options_init (vector_options *options) {
    memset(options, 0, sizeof(vector_options));
    options->v_type = VECTOR_TYPE_F32;
    options->v_distance = VECTOR_DISTANCE_L2;
    options->max_memory = DEFAULT_MAX_MEMORY;
    options->q_type = VECTOR_QUANT_8BIT;
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
    if (rc != SQLITE_OK) printf("Error in vector_serialize_quantization: %s\n", sqlite3_errmsg(db));
    if (vm) sqlite3_finalize(vm);
    return rc;
}

static int vector_rebuild_quantization (sqlite3_context *context, const char *table_name, const char *column_name, table_context *t_ctx, vector_qtype qtype, uint64_t max_memory) {
    
    int rc = SQLITE_NOMEM;
    sqlite3_stmt *vm = NULL;
    char sql[STATIC_SQL_SIZE];
    sqlite3 *db = sqlite3_context_db_handle(context);
    
    const char *pk_name = t_ctx->pk_name;
    int dim = t_ctx->options.v_dim;
    //vector_type type = t_ctx->options.v_type;
    
    // compute size of a single quant, format is: rowid + quantize dimensions
    int q_size = sizeof(int64_t) + (dim * sizeof(uint8_t));
    
    // max_memory == 0 means use all required memory
    if (max_memory == 0) {
        char sql[STATIC_SQL_SIZE];
        sqlite3_snprintf(sizeof(sql), sql, "SELECT COUNT(*) FROM %q;", table_name);
        
        int64_t count = sqlite_read_int64(db, sql);
        max_memory = (count == 0) ? DEFAULT_MAX_MEMORY : (uint64_t)(count * q_size);
    }
    
    // max number of vectors that fits in max_memory
    uint32_t max_vectors = (uint32_t)(max_memory / q_size);
    
    uint8_t *data = sqlite3_malloc64((sqlite3_uint64)(max_vectors * q_size));
    uint8_t *original = data;
    if (!data) goto vector_rebuild_quantization_cleanup;
    
    // SELECT rowid, embedding FROM table
    generate_select_from_table(table_name, column_name, pk_name, sql);
    rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto vector_rebuild_quantization_cleanup;
    
    // STEP 1
    // find global min/max across ALL vectors
    float min_val = MAXFLOAT;
    float max_val = -MAXFLOAT;
    
    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) {rc = SQLITE_OK; break;}
        else if (rc != SQLITE_ROW) break;
        
        float *v = (float *)sqlite3_column_blob(vm, 1);
        for (int i=0; i<dim; ++i) {
            if (v[i] < min_val) min_val = v[i];
            if (v[i] > max_val) max_val = v[i];
        }
    }
    
    // calculate scale and offset and set table them to table context
    float scale = 255.0f / (max_val - min_val);
    float offset = min_val;
    
    t_ctx->scale = scale;
    t_ctx->offset = offset;
    
    // restart processing from the beginning
    rc = sqlite3_reset(vm);
    if (rc != SQLITE_OK) goto vector_rebuild_quantization_cleanup;
    
    // begin quantization (ONLY 8bit is supported in this version)
    uint32_t n_processed = 0;
    uint32_t tot_processed = 0;
    int64_t min_rowid = 0, max_rowid = 0;
    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) {rc = SQLITE_OK; break;}
        else if (rc != SQLITE_ROW) break;
        
        int64_t rowid = (int64_t)sqlite3_column_int64(vm, 0);
        float *v = (float *)sqlite3_column_blob(vm, 1);
        if (n_processed == 0) min_rowid = rowid;
        
        // copy rowid
        INT64_TO_INT8PTR(rowid, data);
        data += sizeof(int64_t);
        
        // quantize vector
        quantize_float32_to_u8(v, data, offset, scale, dim);
        data += (dim * sizeof(uint8_t));
        
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
    if (n_processed > 0) {
        size_t batch_size = data - original;
        rc = vector_serialize_quantization(db, table_name, column_name, n_processed, original, batch_size, min_rowid, max_rowid);
    }
    
vector_rebuild_quantization_cleanup:
    if (rc != SQLITE_OK) printf("Error in vector_rebuild_quantization: %s\n", sqlite3_errmsg(db));
    if (original) sqlite3_free(original);
    if (vm) sqlite3_finalize(vm);
    return rc;
}

static void vector_quantize (sqlite3_context *context, const char *table_name, const char *column_name, const char *arg_options) {
    table_context *t_ctx = vector_context_lookup((vector_context *)sqlite3_user_data(context), table_name, column_name);
    if (!t_ctx) {
        context_result_error(context, SQLITE_ERROR, "Vector context not found for table '%s' and column '%s'. Ensure that vector_init() has been called before using vector_quantize().", table_name, column_name);
        return;
    }
    
    int rc = SQLITE_ERROR;
    char sql[STATIC_SQL_SIZE];
    sqlite3 *db = sqlite3_context_db_handle(context);
    
    rc = sqlite3_exec(db, "BEGIN;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    
    generate_drop_quant_table(table_name, column_name, sql);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    
    generate_create_quant_table(table_name, column_name, sql);
    rc = sqlite3_exec(db, sql, NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    
    vector_options options = vector_options_create();
    bool res = parse_keyvalue_string(context, arg_options, vector_keyvalue_callback, &options);
    if (res == false) return;
    
    rc = vector_rebuild_quantization(context, table_name, column_name, t_ctx, options.q_type, options.max_memory);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    
    rc = sqlite3_exec(db, "COMMIT;", NULL, NULL, NULL);
    if (rc != SQLITE_OK) goto quantize_cleanup;
    
quantize_cleanup:
    if (rc != SQLITE_OK) {
        printf("%s", sqlite3_errmsg(db));
        sqlite3_exec(db, "ROLLBACK;", NULL, NULL, NULL);
        sqlite3_result_error_code(context, rc);
        return;
    }
}

static void vector_quantize3 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT, SQLITE_TEXT};
    if (sanity_check_args(context, "vector_quantize", argc, argv, 3, types) == false) return;
    
    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);
    const char *options = (const char *)sqlite3_value_text(argv[2]);
    vector_quantize(context, table_name, column_name, options);
}

static void vector_quantize2 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT};
    if (sanity_check_args(context, "vector_quantize", argc, argv, 2, types) == false) return;
    
    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);
    vector_quantize(context, table_name, column_name, NULL);
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

static void vector_quantize_preload (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT};
    if (sanity_check_args(context, "vector_quantize_preload", argc, argv, 2, types) == false) return;
    
    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);
    
    vector_context *v_ctx = (vector_context *)sqlite3_user_data(context);
    table_context *t_ctx = vector_context_lookup(v_ctx, table_name, column_name);
    if (!t_ctx) {
        context_result_error(context, SQLITE_ERROR, "Vector context not found for table '%s' and column '%s'. Ensure that vector_init() has been called before using vector_quantize_preload().", table_name, column_name);
        return;
    }
    
    if (t_ctx->preloaded) {
        sqlite3_free(t_ctx->preloaded);
        t_ctx->preloaded = NULL;
        t_ctx->precounter = 0;
    }
    
    char sql[STATIC_SQL_SIZE];
    generate_memory_quant_table(table_name, column_name, sql);
    sqlite3 *db = sqlite3_context_db_handle(context);
    sqlite3_int64 required = sqlite_read_int64(db, sql);
    if (required == 0) {
        context_result_error(context, SQLITE_ERROR, "Unable to read data from database. Ensure that vector_quantize() has been called before using vector_quantize_preload().");
        return;
    }
    
    int counter = 0;
    void *buffer = (void *)sqlite3_malloc64(required);
    if (!buffer) {
        context_result_error(context, SQLITE_NOMEM, "Out of memory: unable to allocate %lld bytes for quant buffer.", (long long)required);
        return;
    }
    
    generate_select_quant_table(table_name, column_name, sql);
    
    int rc = SQLITE_NOMEM;
    sqlite3_stmt *vm = NULL;
    rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto vector_preload_cleanup;
    
    int seek = 0;
    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) {rc = SQLITE_OK; break;} // return error: rebuild must be call (only if first time run)
        else if (rc != SQLITE_ROW) goto vector_preload_cleanup;
        
        int n = sqlite3_column_int(vm, 0);
        int bytes = sqlite3_column_bytes(vm, 1);
        uint8_t *data = (uint8_t *)sqlite3_column_blob(vm, 1);
        
        memcpy(buffer+seek, data, bytes);
        seek += bytes;
        counter += n;
    }
    rc = SQLITE_OK;
    
    t_ctx->preloaded = buffer;
    t_ctx->precounter = counter;
    
vector_preload_cleanup:
    if (rc != SQLITE_OK) printf("Error in vector_quantize_preload: %s\n", sqlite3_errmsg(db));
    if (vm) sqlite3_finalize(vm);
    return;
}

static void vector_cleanup (sqlite3_context *context, int argc, sqlite3_value **argv) {
    int types[] = {SQLITE_TEXT, SQLITE_TEXT};
    if (sanity_check_args(context, "vector_cleanup", argc, argv, 2, types) == false) return;
    
    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);
    
    vector_context *v_ctx = (vector_context *)sqlite3_user_data(context);
    table_context *t_ctx = vector_context_lookup(v_ctx, table_name, column_name);
    if (!t_ctx) return; // if no table context exists then do nothing
    
    // release memory
    if (t_ctx->t_name) sqlite3_free(t_ctx->t_name);
    if (t_ctx->c_name) sqlite3_free(t_ctx->c_name);
    if (t_ctx->pk_name) sqlite3_free(t_ctx->pk_name);
    if (t_ctx->preloaded) sqlite3_free(t_ctx->preloaded);
    memset(t_ctx, 0, sizeof(table_context));
    
    // drop quant table (if any)
    char sql[STATIC_SQL_SIZE];
    sqlite3 *db = sqlite3_context_db_handle(context);
    generate_drop_quant_table(table_name, column_name, sql);
    sqlite3_exec(db, sql, NULL, NULL, NULL);
    
    // do not decrease v_ctx->table_count
}

// MARK: -

static char *vector_convert_from_json (sqlite3_context *context, vector_type type, const char *json, int *size) {
    char *blob = NULL;
    
    // skip leading whitespace
    SKIP_SPACES(json);
    
    // sanity check the JSON start array character
    if (*json != '[') {
        context_result_error(context, SQLITE_ERROR, "Malformed JSON: expected '[' at the beginning of the array.");
        return NULL;
    }
    json++;

    
    // count number of commas to estimate (over-estimate) the number of entries in the json
    int estimated_count = 0;
    for (const char *p = json; *p; ++p) {
        if (*p == ',') estimated_count++;
    }
    
    // allocate blob
    size_t item_size = vector_type_to_size(type);
    size_t alloc = (estimated_count + 1) * item_size;
    blob = sqlite3_malloc((int)alloc);
    if (!blob) {
        context_result_error(context, SQLITE_NOMEM, "Out of memory: unable to allocate %zu bytes for BLOB buffer.", alloc);
        return NULL;
    }
    
    float *float_blob = (float *)blob;
    //uint16_t *uint16_blob = (uint16_t *)blob;
    //bfloat16 *bfloat16_blob = (bfloat16 *)blob;
    uint8_t *uint8_blob = (uint8_t *)blob;
    int8_t *int8_blob = (int8_t *)blob;
    
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
        if (p == endptr) {
            // parsing failed
            context_result_error(context, SQLITE_ERROR, "Malformed JSON: expected a number at position %d (found '%c').", (int)(p - json) + 1, *p ? *p : '?');
            sqlite3_free(blob);
            return NULL;
        }
        
        switch (type) {
            case VECTOR_TYPE_F32:
                float_blob[count++] = (float)value;
                break;
                //case VECTOR_TYPE_F16:
                //uint16_blob[count++] = fp16_ieee_from_fp32_value((float)value);
                //break;
                //case VECTOR_TYPE_BF16:
                //bfloat16_blob[count++] = float32_to_bfloat16((float)value);
                //break;
            case VECTOR_TYPE_U8:
                uint8_blob[count++] = (uint8_t)value;
                break;
            case VECTOR_TYPE_I8:
                int8_blob[count++] = (int8_t)value;
                break;
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
            // unexpected character
            context_result_error(context, SQLITE_ERROR, "Malformed JSON: unexpected character '%c' at position %d.", *p ? *p : '?', (int)(p - json) + 1);
            sqlite3_free(blob);
            return NULL;
        }
    }
    
    if (size) *size = (int)(count * item_size);
    return blob;
}

static void vector_convert (sqlite3_context *context, vector_type type, int argc, sqlite3_value **argv) {
    sqlite3_value *value = argv[0];
    int value_size = sqlite3_value_bytes(value);
    int value_type = sqlite3_value_type(value);
    
    if (value_type == SQLITE_BLOB) {
        // the only check we can perform is that the blob size is an exact multiplier of the vector type
        if (value_size % vector_type_to_size(type) != 0) {
            context_result_error(context, SQLITE_ERROR, "Invalid BLOB size for format '%s': size must be a multiple of %d bytes.", vector_type_to_name(type), vector_type_to_size(type));
            return;
        }
        sqlite3_result_value(context, value);
        return;
    }
    
    if (value_type == SQLITE_TEXT) {
        // try to parse JSON array value
        char *blob = vector_convert_from_json(context, type, (const char *)sqlite3_value_text(value), &value_size);
        if (!blob) return; // error is set in the context
        
        sqlite3_result_blob(context, (const void *)blob, value_size, sqlite3_free);
        return;
    }
    
    context_result_error(context, SQLITE_ERROR, "Unsupported input type: only BLOB and TEXT values are accepted (received %s).", sqlite_type_name(value_type));
}

static void vector_convert_f32 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    vector_convert(context, VECTOR_TYPE_F32, argc, argv);
}

static void vector_convert_f16 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    //vector_convert(context, VECTOR_TYPE_F16, argc, argv);
}

static void vector_convert_bf16 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    //vector_convert(context, VECTOR_TYPE_BF16, argc, argv);
}

static void vector_convert_u8 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    vector_convert(context, VECTOR_TYPE_U8, argc, argv);
}

static void vector_convert_i8 (sqlite3_context *context, int argc, sqlite3_value **argv) {
    vector_convert(context, VECTOR_TYPE_I8, argc, argv);
}

// MARK: - Modules -

static int vCursorFilterCommon (sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr, int argc, sqlite3_value **argv, const char *fname, vcursor_run_callback run_callback, vcursor_sort_callback sort_callback) {
    
    vFullScanCursor *c = (vFullScanCursor *)cur;
    vFullScan *vtab = (vFullScan *)cur->pVtab;
    
    // sanity check arguments
    if (argc != 4) {
        return sqlite_vtab_set_error(&vtab->base, "%s expects %d arguments, but %d were provided.", fname, 4, argc);
    }
    
    // SQLITE_TEXT, SQLITE_TEXT, SQLITE_TEXT or SQLITE_BLOB, SQLITE_INTEGER
    for (int i=0; i<argc; ++i) {
        int actual_type = sqlite3_value_type(argv[i]);
        switch (i) {
            case 0:
            case 1:
                if (actual_type != SQLITE_TEXT)
                    return sqlite_vtab_set_error(&vtab->base, "%s: argument %d must be of type TEXT (got %s).", fname, (i+1), argc, sqlite_type_name(actual_type));
                break;
            case 2:
                if ((actual_type != SQLITE_TEXT) && (actual_type != SQLITE_BLOB))
                    return sqlite_vtab_set_error(&vtab->base, "%s: argument %d must be of type TEXT or BLOB (got %s).", fname, (i+1), argc, sqlite_type_name(actual_type));
                break;
            case 3:
                if (actual_type != SQLITE_INTEGER)
                    return sqlite_vtab_set_error(&vtab->base, "%s: argument %d must be of type INTEGER (got %s).", fname, (i+1), argc, sqlite_type_name(actual_type));
                break;
        }
    }
    
    // retrieve arguments
    const char *table_name = (const char *)sqlite3_value_text(argv[0]);
    const char *column_name = (const char *)sqlite3_value_text(argv[1]);
    table_context *t_ctx = vector_context_lookup(vtab->ctx, table_name, column_name);
    if (!t_ctx) {
        return sqlite_vtab_set_error(&vtab->base, "%s: unable to retrieve context.", fname);
    }
    
    const void *vector = NULL;
    int vsize = 0;
    if (sqlite3_value_type(argv[2]) == SQLITE_TEXT) {
        vsize = sqlite3_value_bytes(argv[2]);
        vector = (const void *)vector_convert_from_json(NULL, t_ctx->options.v_type, (const char *)sqlite3_value_text(argv[2]), &vsize);
        // TODO: error here
        if (!vector) return SQLITE_ERROR;
    } else {
        vector = (const void *)sqlite3_value_blob(argv[2]);
        vsize = sqlite3_value_bytes(argv[2]);
    }
    
    int k = sqlite3_value_int(argv[3]);
    
    // nothing needs to be returned
    if (k == 0) return SQLITE_DONE;
    
    c->rowids = (sqlite3_int64 *)sqlite3_malloc(k * sizeof(sqlite3_int64));
    if (c->rowids == NULL) return SQLITE_NOMEM;
    memset(c->rowids, 0, k*sizeof(sqlite3_int64));
    
    c->distance = (double *)sqlite3_malloc(k * sizeof(double));
    if (c->distance == NULL) return SQLITE_NOMEM;
    for (int i=0; i<k; ++i) c->distance[i] = INFINITY;
    
    c->size = 0;
    c->table = t_ctx;
    c->row_index = 0;
    c->row_count = k;
    
    int rc = run_callback(vtab->db, c, vector, vsize);
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
    pIdxInfo->estimatedCost = (double)1;
    pIdxInfo->estimatedRows = 100;
    pIdxInfo->orderByConsumed = 1;
    pIdxInfo->idxNum = 1;
    
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
                pIdxInfo->aConstraintUsage[i].argvIndex = 4;
                pIdxInfo->aConstraintUsage[i].omit = 1;
                break;
        }
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
    sqlite3_free(c);
    return SQLITE_OK;
}

static int vFullScanCursorNext (sqlite3_vtab_cursor *cur){
    vFullScanCursor *c = (vFullScanCursor *)cur;
    c->row_index++;
    return SQLITE_OK;
}

static int vFullScanCursorEof (sqlite3_vtab_cursor *cur){
    vFullScanCursor *c = (vFullScanCursor *)cur;
    return (c->row_index == c->row_count);
}

static int vFullScanCursorColumn (sqlite3_vtab_cursor *cur, sqlite3_context *context, int iCol) {
    vFullScanCursor *c = (vFullScanCursor *)cur;
    if (iCol == VECTOR_COLUMN_ROWID) {
        sqlite3_result_int64(context, c->rowids[c->row_index]);
    } else if (iCol == VECTOR_COLUMN_DISTANCE) {
        sqlite3_result_double(context, c->distance[c->row_index]);
    }
    return SQLITE_OK;
}

static int vFullScanCursorRowid (sqlite3_vtab_cursor *cur, sqlite_int64 *pRowid) {
    vFullScanCursor *c = (vFullScanCursor *)cur;
    *pRowid = c->rowids[c->row_index];
    return SQLITE_OK;
}

static inline int vFullScanFindMaxIndex (double *values, int n) {
    int max_idx = 0;
    if (n <= 32) {
        // use simple version
        for (int i = 1; i < n; ++i) {
            if (values[i] > values[max_idx]) {max_idx = i;}
        }
        return max_idx;
    }
    
    // use unrolled version
    double max_val = values[0];
    int i = 1;
    
    // unroll loop in blocks of 4
    for (; i + 3 < n; i += 4) {
        if (values[i] > max_val) {max_val = values[i]; max_idx = i;}
        if (values[i + 1] > max_val) {max_val = values[i + 1]; max_idx = i + 1;}
        if (values[i + 2] > max_val) {max_val = values[i + 2]; max_idx = i + 2;}
        if (values[i + 3] > max_val) {max_val = values[i + 3]; max_idx = i + 3;}
    }
    
    // process remaining elements
    for (; i < n; ++i) {
        if (values[i] > max_val) {max_val = values[i]; max_idx = i;}
    }
    return max_idx;
}

static int vFullScanSortSlots (vFullScanCursor *c) {
    int    counter = 0;
    int    row_count = c->row_count;
    double *distance = c->distance;
    sqlite3_int64 *rowids = c->rowids;
    
    for (int i = 0; i < row_count - 1; ++i) {
        if (distance[i] == INFINITY) ++counter;
        for (int j = i + 1; j < row_count; ++j) {
            if (distance[j] < distance[i]) {
                SWAP(double, distance[i], distance[j]);
                SWAP(sqlite3_int64, rowids[i], rowids[j]);
            }
        }
    }
    
    if (distance[row_count-1] == INFINITY) ++counter;
    return counter;
}

static int vFullScanRun (sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size) {
    const char *pk_name = c->table->pk_name;
    const char *col_name = c->table->c_name;
    const char *table_name = c->table->t_name;
    int dimension = c->table->options.v_dim;
    
    char *sql = sqlite3_mprintf("SELECT %q, %q FROM %q;", pk_name, col_name, table_name);
    if (!sql) return SQLITE_NOMEM;
    
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    // compute distance function
    vector_distance vd = c->table->options.v_distance;
    vector_type vt = c->table->options.v_type;
    distance_function_t distance_fn = dispatch_distance_table[vd][vt];
    
    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) {rc = SQLITE_OK; goto cleanup;}
        if (rc != SQLITE_ROW) goto cleanup;
        
        float *v2 = (float *)sqlite3_column_blob(vm, 1);
        double distance = distance_fn((const void *)v1, (const void *)v2, dimension);
        
        if (distance < c->distance[c->max_index]) {
            c->distance[c->max_index] = distance;
            c->rowids[c->max_index] = sqlite3_column_int64(vm, 0);
            c->max_index = vFullScanFindMaxIndex(c->distance, c->row_count);
        }
    }
    
cleanup:
    if (sql) sqlite3_free(sql);
    if (vm) sqlite3_finalize(vm);
    return rc;
}

static int vFullScanCursorFilter (sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    return vCursorFilterCommon(cur, idxNum, idxStr, argc, argv, "vector_full_scan", vFullScanRun, vFullScanSortSlots);
}

// MARK: -

static int vQuantRunMemory(vFullScanCursor *c, uint8_t *v, int dim) {
    const int counter = c->table->precounter;
    const uint8_t *data = c->table->preloaded;
    const size_t rowid_size = sizeof(int64_t);
    const size_t vector_size = dim * sizeof(uint8_t);
    const size_t total_stride = rowid_size + vector_size;

    double *distance = c->distance;
    int64_t *rowids = c->rowids;
    int max_index = c->max_index;
    double current_max = distance[max_index];
    
    // compute distance function
    vector_distance vd = c->table->options.v_distance;
    vector_type vt = VECTOR_TYPE_U8;
    distance_function_t distance_fn = dispatch_distance_table[vd][vt];
    
    for (int i = 0; i < counter; ++i) {
        const uint8_t *current_data = data + (i * total_stride);
        const uint8_t *vector_data = current_data + rowid_size;

        float dist = distance_fn((const void *)v, (const void *)vector_data, dim);

        if (dist < current_max) {
            distance[max_index] = dist;
            rowids[max_index] = INT64_FROM_INT8PTR(current_data);

            // Recompute max index efficiently
            max_index = vFullScanFindMaxIndex(distance, c->row_count);
            current_max = distance[max_index];
        }
    }

    c->max_index = max_index;
    return SQLITE_OK;
}

static int vQuantRun (sqlite3 *db, vFullScanCursor *c, const void *v1, int v1size) {
    // quantize target vector
    int dimension = c->table->options.v_dim;
    uint8_t *v = (uint8_t *)sqlite3_malloc(dimension * sizeof(int8_t));
    if (!v) return SQLITE_NOMEM;
    
    quantize_float32_to_u8((float *)v1, v, c->table->offset, c->table->scale, dimension);
    if (c->table->preloaded) return vQuantRunMemory(c, v, dimension);
    
    char sql[STATIC_SQL_SIZE];
    generate_select_quant_table(c->table->t_name, c->table->c_name, sql);
    sqlite3_stmt *vm = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &vm, NULL);
    if (rc != SQLITE_OK) goto kann_run_cleanup;
    
    // precompute constants
    const size_t rowid_size = sizeof(int64_t);
    const size_t vector_size = dimension * sizeof(uint8_t);
    const size_t total_stride = rowid_size + vector_size;
    
    // compute distance function
    vector_distance vd = c->table->options.v_distance;
    vector_type vt = VECTOR_TYPE_U8;
    distance_function_t distance_fn = dispatch_distance_table[vd][vt];
    
    while (1) {
        rc = sqlite3_step(vm);
        if (rc == SQLITE_DONE) {rc = SQLITE_OK; break;} // return error: rebuild must be call (only if first time run)
        else if (rc != SQLITE_ROW) goto kann_run_cleanup;
        
        int counter = sqlite3_column_int(vm, 0);
        uint8_t *data = (uint8_t *)sqlite3_column_blob(vm, 1);
        
        // cache the maximum value to avoid repeated memory accesses
        double current_max_distance = c->distance[c->max_index];
        
        for (int i=0; i<counter; ++i) {
            const uint8_t *current_data = data + (i * total_stride);
            const uint8_t *vector_data = current_data + rowid_size;
            double distance = (double)distance_fn((const void *)v, (const void *)vector_data, dimension);
            
            if (distance < current_max_distance) {
                c->distance[c->max_index] = distance;
                c->rowids[c->max_index] = INT64_FROM_INT8PTR(current_data);
                c->max_index = vFullScanFindMaxIndex(c->distance, c->row_count);
                current_max_distance = c->distance[c->max_index]; // update cached max
            }
        }
    }
    
    rc = SQLITE_OK;
    
kann_run_cleanup:
    if (rc != SQLITE_OK) printf("Error in vector_rebuild_quantization: %s\n", sqlite3_errmsg(db));
    if (vm) sqlite3_finalize(vm);
    return rc;
}


static int vQuantCursorFilter (sqlite3_vtab_cursor *cur, int idxNum, const char *idxStr, int argc, sqlite3_value **argv) {
    return vCursorFilterCommon(cur, idxNum, idxStr, argc, argv, "vector_quantize_scan", vQuantRun, vFullScanSortSlots);
}
 
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

static sqlite3_module vQuantModule = {
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
    
    if ((options.v_type == VECTOR_TYPE_F16) || (options.v_type == VECTOR_TYPE_BF16)) {
        context_result_error(context, SQLITE_ERROR, "FLOAT16 and FLOATB16 vector types are currently under development. Please update to the latest version.");
        return;
    }
    
    // check if table is already loaded
    vector_context *v_ctx = (vector_context *)sqlite3_user_data(context);
    table_context *t_ctx = vector_context_lookup(v_ctx, table_name, column_name);
    if (t_ctx) {
        // sanity check
        if (options.v_dim != t_ctx->options.v_dim) {
            context_result_error(context, SQLITE_ERROR, "Inconsistent vector dimension for '%s.%s': existing=%d, provided=%d.", table_name, column_name, t_ctx->options.v_dim, options.v_dim);
            return;
        }
        
        if (options.v_type != t_ctx->options.v_type) {
            context_result_error(context, SQLITE_ERROR, "Inconsistent vector type for '%s.%s': existing=%s, provided=%s.", table_name, column_name, vector_type_to_name(t_ctx->options.v_type), vector_type_to_name(options.v_type));
            return;
        }
        
        if (options.v_normalized != t_ctx->options.v_normalized) {
            context_result_error(context, SQLITE_ERROR, "Inconsistent normalization flag for '%s.%s': existing=%s, provided=%s.", table_name, column_name, t_ctx->options.v_normalized ? "true" : "false", options.v_normalized ? "true" : "false");
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
    
// MARK: -

SQLITE_VECTOR_API int sqlite3_vector_init (sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi) {
    int rc = SQLITE_OK;
    
    init_distance_functions(false);
    
    // init context
    void *ctx = vector_context_create();
    if (!ctx) {
        if (pzErrMsg) *pzErrMsg = sqlite3_mprintf("Out of memory: failed to allocate vector extension context.");
        return SQLITE_NOMEM;
    }
    
    rc = sqlite3_create_function(db, "vector_version", 0, SQLITE_UTF8, ctx, vector_version, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_function(db, "vector_backend", 0, SQLITE_UTF8, ctx, vector_backend, NULL, NULL);
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
    rc = sqlite3_create_function(db, "vector_cleanup", 2, SQLITE_UTF8, ctx, vector_cleanup, NULL, NULL);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_module(db, "vector_full_scan", &vFullScanModule, ctx);
    if (rc != SQLITE_OK) goto cleanup;
    
    rc = sqlite3_create_module(db, "vector_quantize_scan", &vQuantModule, ctx);
    if (rc != SQLITE_OK) goto cleanup;
    
cleanup:
    return rc;
}

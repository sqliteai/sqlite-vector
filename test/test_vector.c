/*
 * test_vector.c
 * Comprehensive unit tests for the SQLite Vector extension.
 *
 * Compiled with -DSQLITE_CORE so sqlite3_vector_init links statically.
 * Usage: gcc -DSQLITE_CORE ... -o test_vector && ./test_vector
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <strings.h>
#include <math.h>
#include <unistd.h>
#include "sqlite3.h"
#include "sqlite-vector.h"

/* ---------- Test infrastructure ---------- */

static int failures = 0;
static int passes   = 0;

#define ASSERT(cond, msg) do {                        \
    if (!(cond)) { printf("FAIL: %s\n", msg); failures++; } \
    else         { printf("PASS: %s\n", msg); passes++;    } \
} while (0)

/* Execute SQL that must succeed; returns SQLITE_OK or aborts the test. */
static int exec_sql(sqlite3 *db, const char *sql) {
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
    if (rc != SQLITE_OK) {
        printf("  SQL error (%d): %s\n  Statement: %s\n", rc, err ? err : "unknown", sql);
        sqlite3_free(err);
    }
    return rc;
}

/* ---------- Helper: create, populate, and init a vector table ---------- */

/*
 * Sets up a table named `tbl` with columns (id INTEGER PRIMARY KEY, v BLOB),
 * inserts `n` vectors of the given type converted from JSON via vector_as_<type>(),
 * and calls vector_init() with the specified type, distance, and dimension.
 *
 * `vecs` is an array of JSON strings, e.g. "[1.0, 2.0, 3.0]".
 */
static int setup_table(sqlite3 *db, const char *tbl, const char *type,
                       const char *distance, int dim,
                       const char **vecs, int n) {
    char sql[2048];

    /* Create table */
    snprintf(sql, sizeof(sql), "CREATE TABLE \"%s\" (id INTEGER PRIMARY KEY, v BLOB);", tbl);
    if (exec_sql(db, sql) != SQLITE_OK) return -1;

    /* Insert vectors */
    for (int i = 0; i < n; i++) {
        snprintf(sql, sizeof(sql),
                 "INSERT INTO \"%s\" (id, v) VALUES (%d, vector_as_%s('%s'));",
                 tbl, i + 1, type, vecs[i]);
        if (exec_sql(db, sql) != SQLITE_OK) return -1;
    }

    /* vector_init */
    snprintf(sql, sizeof(sql),
             "SELECT vector_init('%s', 'v', 'type=%s,dimension=%d,distance=%s');",
             tbl, type, dim, distance);
    if (exec_sql(db, sql) != SQLITE_OK) return -1;

    return 0;
}

static int setup_f32_blob_table(sqlite3 *db, const char *tbl, const char *distance,
                                int dim, const float *rows, int n) {
    char sql[512];
    sqlite3_stmt *stmt = NULL;

    snprintf(sql, sizeof(sql), "CREATE TABLE \"%s\" (id INTEGER PRIMARY KEY, v BLOB);", tbl);
    if (exec_sql(db, sql) != SQLITE_OK) return -1;

    snprintf(sql, sizeof(sql), "INSERT INTO \"%s\" (id, v) VALUES (?1, ?2);", tbl);
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return -1;

    for (int i = 0; i < n; ++i) {
        sqlite3_reset(stmt);
        sqlite3_clear_bindings(stmt);
        sqlite3_bind_int(stmt, 1, i + 1);
        sqlite3_bind_blob(stmt, 2, rows + (size_t)i * dim, dim * (int)sizeof(float), SQLITE_STATIC);
        rc = sqlite3_step(stmt);
        if (rc != SQLITE_DONE) {
            sqlite3_finalize(stmt);
            return -1;
        }
    }
    sqlite3_finalize(stmt);

    snprintf(sql, sizeof(sql),
             "SELECT vector_init('%s', 'v', 'type=f32,dimension=%d,distance=%s');",
             tbl, dim, distance);
    return exec_sql(db, sql) == SQLITE_OK ? 0 : -1;
}

/* ---------- Callback helpers for querying results ---------- */

typedef struct {
    int    count;
    int    ids[64];
    double distances[64];
} scan_result;

static int scan_cb(void *ctx, int ncols, char **vals, char **names) {
    (void)names;
    scan_result *r = (scan_result *)ctx;
    if (r->count < 64 && ncols >= 2 && vals[1]) {
        r->ids[r->count] = vals[0] ? atoi(vals[0]) : 0;
        r->distances[r->count] = atof(vals[1]);
    }
    r->count++;
    return 0;
}

/* Like scan_cb but reads distance from column 0 (for queries where distance is first). */
static int scan_cb_col0(void *ctx, int ncols, char **vals, char **names) {
    (void)names;
    scan_result *r = (scan_result *)ctx;
    if (r->count < 64 && ncols >= 1 && vals[0]) {
        r->distances[r->count] = atof(vals[0]);
    }
    r->count++;
    return 0;
}

static int exec_scan_sql(sqlite3 *db, const char *sql, sqlite3_callback cb, void *ctx) {
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, cb, ctx, &err);
    if (rc != SQLITE_OK) {
        printf("  SQL error (%d): %s\n  Statement: %s\n", rc, err ? err : "unknown", sql);
        sqlite3_free(err);
    }
    return rc;
}

static int exec_bound_scan_sql(sqlite3 *db, const char *sql, const float *query, int dim, scan_result *r) {
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return rc;

    rc = sqlite3_bind_blob(stmt, 1, query, dim * (int)sizeof(float), SQLITE_STATIC);
    if (rc == SQLITE_OK) {
        while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
            if (r->count < 64) {
                r->ids[r->count] = sqlite3_column_int(stmt, 0);
                r->distances[r->count] = sqlite3_column_double(stmt, 1);
            }
            r->count++;
        }
        if (rc == SQLITE_DONE) rc = SQLITE_OK;
    }

    int frc = sqlite3_finalize(stmt);
    return rc == SQLITE_OK ? frc : rc;
}

static int open_stmt_count(sqlite3 *db) {
    int count = 0;
    for (sqlite3_stmt *stmt = sqlite3_next_stmt(db, NULL); stmt; stmt = sqlite3_next_stmt(db, stmt)) {
        count++;
    }
    return count;
}

/* ---------- Test: basics ---------- */

static void test_basics(sqlite3 *db) {
    printf("\n=== Basics ===\n");

    /* vector_version() */
    {
        sqlite3_stmt *stmt;
        int rc = sqlite3_prepare_v2(db, "SELECT vector_version();", -1, &stmt, NULL);
        ASSERT(rc == SQLITE_OK, "vector_version() prepares");
        if (rc == SQLITE_OK) {
            rc = sqlite3_step(stmt);
            ASSERT(rc == SQLITE_ROW, "vector_version() returns a row");
            const char *v = (const char *)sqlite3_column_text(stmt, 0);
            ASSERT(v != NULL && strlen(v) > 0, "vector_version() returns non-empty text");
        }
        sqlite3_finalize(stmt);
    }

    /* vector_backend() */
    {
        sqlite3_stmt *stmt;
        int rc = sqlite3_prepare_v2(db, "SELECT vector_backend();", -1, &stmt, NULL);
        ASSERT(rc == SQLITE_OK, "vector_backend() prepares");
        if (rc == SQLITE_OK) {
            rc = sqlite3_step(stmt);
            ASSERT(rc == SQLITE_ROW, "vector_backend() returns a row");
            const char *v = (const char *)sqlite3_column_text(stmt, 0);
            ASSERT(v != NULL && strlen(v) > 0, "vector_backend() returns non-empty text");
        }
        sqlite3_finalize(stmt);
    }

    /* vector_turboquant_backend() */
    {
        sqlite3_stmt *stmt;
        int rc = sqlite3_prepare_v2(db, "SELECT vector_turboquant_backend();", -1, &stmt, NULL);
        ASSERT(rc == SQLITE_OK, "vector_turboquant_backend() prepares");
        if (rc == SQLITE_OK) {
            rc = sqlite3_step(stmt);
            ASSERT(rc == SQLITE_ROW, "vector_turboquant_backend() returns a row");
            const char *v = (const char *)sqlite3_column_text(stmt, 0);
            ASSERT(v != NULL && strlen(v) > 0, "vector_turboquant_backend() returns non-empty text");
        }
        sqlite3_finalize(stmt);
    }
}

/* ---------- Test: vector_full_scan for a given (type, distance) pair ---------- */

static void test_full_scan(sqlite3 *db, const char *type, const char *distance,
                           int dim, const char **vecs, int nvecs,
                           const char *query_vec) {
    char tbl[64], sql[1024], msg[256];
    snprintf(tbl, sizeof(tbl), "tfs_%s_%s", type, distance);

    /* lowercase table name for uniqueness */
    for (char *p = tbl; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;

    if (setup_table(db, tbl, type, distance, dim, vecs, nvecs) != 0) {
        snprintf(msg, sizeof(msg), "full_scan setup %s/%s", type, distance);
        ASSERT(0, msg);
        return;
    }

    /* DOT distance returns negative dot product, so skip non-negative checks */
    int is_dot = (strcasecmp(distance, "DOT") == 0);

    /* Top-k mode (k=3) */
    {
        scan_result r = {0};
        snprintf(sql, sizeof(sql),
                 "SELECT id, distance FROM vector_full_scan('%s', 'v', vector_as_%s('%s'), 3);",
                 tbl, type, query_vec);
        char *err = NULL;
        int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
        snprintf(msg, sizeof(msg), "full_scan top-k executes (%s/%s)", type, distance);
        ASSERT(rc == SQLITE_OK, msg);
        if (err) { printf("  err: %s\n", err); sqlite3_free(err); }

        snprintf(msg, sizeof(msg), "full_scan top-k returns 3 rows (%s/%s)", type, distance);
        ASSERT(r.count == 3, msg);

        /* Distances should be non-negative (DOT returns negative dot product, so skip) */
        if (!is_dot) {
            int all_non_neg = 1;
            for (int i = 0; i < r.count; i++) {
                if (r.distances[i] < 0) all_non_neg = 0;
            }
            snprintf(msg, sizeof(msg), "full_scan top-k distances >= 0 (%s/%s)", type, distance);
            ASSERT(all_non_neg, msg);
        }

        /* Distances should be sorted ascending */
        int sorted = 1;
        for (int i = 1; i < r.count; i++) {
            if (r.distances[i] < r.distances[i - 1]) sorted = 0;
        }
        snprintf(msg, sizeof(msg), "full_scan top-k distances sorted (%s/%s)", type, distance);
        ASSERT(sorted, msg);
    }

    /* Streaming mode (no k, use LIMIT) */
    {
        scan_result r = {0};
        snprintf(sql, sizeof(sql),
                 "SELECT id, distance FROM vector_full_scan('%s', 'v', vector_as_%s('%s')) LIMIT 5;",
                 tbl, type, query_vec);
        char *err = NULL;
        int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
        snprintf(msg, sizeof(msg), "full_scan stream executes (%s/%s)", type, distance);
        ASSERT(rc == SQLITE_OK, msg);
        if (err) { printf("  err: %s\n", err); sqlite3_free(err); }

        snprintf(msg, sizeof(msg), "full_scan stream returns rows (%s/%s)", type, distance);
        ASSERT(r.count > 0, msg);

        if (!is_dot) {
            int all_non_neg = 1;
            for (int i = 0; i < r.count; i++) {
                if (r.distances[i] < 0) all_non_neg = 0;
            }
            snprintf(msg, sizeof(msg), "full_scan stream distances >= 0 (%s/%s)", type, distance);
            ASSERT(all_non_neg, msg);
        }
    }
}

/* ---------- Test: vector_quantize_scan for a given (type, qtype) pair ---------- */

static void test_quantize_scan(sqlite3 *db, const char *type, const char *qtype,
                               int dim, const char **vecs, int nvecs,
                               const char *query_vec) {
    char tbl[64], sql[1024], msg[256];
    snprintf(tbl, sizeof(tbl), "tqs_%s_%s", type, qtype);

    for (char *p = tbl; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;

    /* Use L2 distance (or HAMMING for BIT) */
    const char *distance = (strcasecmp(type, "BIT") == 0) ? "HAMMING" : "L2";

    if (setup_table(db, tbl, type, distance, dim, vecs, nvecs) != 0) {
        snprintf(msg, sizeof(msg), "quantize_scan setup %s/%s", type, qtype);
        ASSERT(0, msg);
        return;
    }

    /* vector_quantize */
    snprintf(sql, sizeof(sql),
             "SELECT vector_quantize('%s', 'v', 'qtype=%s');", tbl, qtype);
    if (exec_sql(db, sql) != SQLITE_OK) {
        snprintf(msg, sizeof(msg), "vector_quantize %s/%s", type, qtype);
        ASSERT(0, msg);
        return;
    }

    /* Top-k mode */
    {
        scan_result r = {0};
        snprintf(sql, sizeof(sql),
                 "SELECT id, distance FROM vector_quantize_scan('%s', 'v', vector_as_%s('%s'), 3);",
                 tbl, type, query_vec);
        char *err = NULL;
        int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
        snprintf(msg, sizeof(msg), "quantize_scan top-k executes (%s/%s)", type, qtype);
        ASSERT(rc == SQLITE_OK, msg);
        if (err) { printf("  err: %s\n", err); sqlite3_free(err); }

        snprintf(msg, sizeof(msg), "quantize_scan top-k returns rows (%s/%s)", type, qtype);
        ASSERT(r.count > 0, msg);
    }

    /* Streaming mode */
    {
        scan_result r = {0};
        snprintf(sql, sizeof(sql),
                 "SELECT id, distance FROM vector_quantize_scan('%s', 'v', vector_as_%s('%s')) LIMIT 5;",
                 tbl, type, query_vec);
        char *err = NULL;
        int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
        snprintf(msg, sizeof(msg), "quantize_scan stream executes (%s/%s)", type, qtype);
        ASSERT(rc == SQLITE_OK, msg);
        if (err) { printf("  err: %s\n", err); sqlite3_free(err); }

        snprintf(msg, sizeof(msg), "quantize_scan stream returns rows (%s/%s)", type, qtype);
        ASSERT(r.count > 0, msg);
    }
}

static void test_turboquant(sqlite3 *db) {
    printf("\n=== TurboQuant ===\n");

    char primary_tbl[64] = "tq_f32_DOT_4";
    const char *tbl = primary_tbl;
    const int dim = 16;
    const char *vecs[] = {
        "[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]",
        "[-1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0]",
        "[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1]",
        "[0, 0, 0, 0, 0, 0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1]",
        "[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]",
        "[0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]",
        "[2, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]",
        "[0, 0, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0]"
    };
    const char *query = "[1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]";
    char sql[1024];

    ASSERT(open_stmt_count(db) == 0, "no open statements before TurboQuant test");

    const char *distances[] = {"DOT", "COSINE", "L2", "SQUARED_L2"};
    const int bits_values[] = {2, 3, 4};
    for (int d = 0; d < 4; ++d) {
        for (int b = 0; b < 3; ++b) {
            int bits = bits_values[b];
            char tname[64];
            char msg[160];
            snprintf(tname, sizeof(tname), "tq_f32_%s_%d", distances[d], bits);
            if (setup_table(db, tname, "f32", distances[d], dim, vecs, 8) != 0) {
                snprintf(msg, sizeof(msg), "TurboQuant setup %s qbits=%d", distances[d], bits);
                ASSERT(0, msg);
                return;
            }

            snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s', 'v', 'qtype=TURBO,qbits=%d,max_memory=96');", tname, bits);
            snprintf(msg, sizeof(msg), "vector_quantize TurboQuant %s qbits=%d executes", distances[d], bits);
            ASSERT(exec_sql(db, sql) == SQLITE_OK, msg);

            {
                sqlite3_stmt *stmt = NULL;
                snprintf(sql, sizeof(sql), "SELECT vector_quantize_memory('%s', 'v');", tname);
                int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
                ASSERT(rc == SQLITE_OK, "TurboQuant memory prepares");
                if (rc == SQLITE_OK && sqlite3_step(stmt) == SQLITE_ROW) {
                    sqlite3_int64 bytes = sqlite3_column_int64(stmt, 0);
                    sqlite3_int64 expected = (sqlite3_int64)8 * (sqlite3_int64)(sizeof(int64_t) + sizeof(float) + ((dim * bits + 7) / 8));
                    snprintf(msg, sizeof(msg), "TurboQuant qbits=%d uses expected compact storage", bits);
                    ASSERT(bytes == expected, msg);
                } else {
                    ASSERT(0, "TurboQuant memory returns a row");
                }
                sqlite3_finalize(stmt);
            }

            scan_result exact = {0};
            snprintf(sql, sizeof(sql),
                     "SELECT id, distance FROM vector_full_scan('%s', 'v', vector_as_f32('%s'), 1);",
                     tname, query);
            snprintf(msg, sizeof(msg), "TurboQuant exact top-1 executes %s qbits=%d", distances[d], bits);
            ASSERT(sqlite3_exec(db, sql, scan_cb, &exact, NULL) == SQLITE_OK, msg);

            scan_result approx = {0};
            snprintf(sql, sizeof(sql),
                     "SELECT id, distance FROM vector_quantize_scan('%s', 'v', vector_as_f32('%s'), 1);",
                     tname, query);
            snprintf(msg, sizeof(msg), "TurboQuant top-1 executes %s qbits=%d", distances[d], bits);
            ASSERT(sqlite3_exec(db, sql, scan_cb, &approx, NULL) == SQLITE_OK, msg);
            snprintf(msg, sizeof(msg), "TurboQuant top-1 matches full scan %s qbits=%d", distances[d], bits);
            ASSERT(exact.count == 1 && approx.count == 1 && exact.ids[0] == approx.ids[0], msg);
            if (strcmp(distances[d], "DOT") == 0) {
                ASSERT(approx.distances[0] < 0.0, "TurboQuant DOT distance uses negative score convention");
            }

            if (strcmp(distances[d], "DOT") == 0 && bits == 4) {
                snprintf(primary_tbl, sizeof(primary_tbl), "%s", tname);
                tbl = primary_tbl;
                snprintf(sql, sizeof(sql), "SELECT vector_quantize_preload('%s', 'v');", tbl);
                ASSERT(exec_sql(db, sql) == SQLITE_OK, "TurboQuant preload executes");

                memset(&approx, 0, sizeof(approx));
                snprintf(sql, sizeof(sql),
                         "SELECT id, distance FROM vector_quantize_scan('%s', 'v', vector_as_f32('%s'), 3);",
                         tbl, query);
                ASSERT(exec_scan_sql(db, sql, scan_cb, &approx) == SQLITE_OK, "preloaded TurboQuant top-k executes");
                ASSERT(approx.count == 3, "preloaded TurboQuant returns k rows");

                scan_result streamed = {0};
                snprintf(sql, sizeof(sql),
                         "SELECT id, distance FROM vector_quantize_scan('%s', 'v', vector_as_f32('%s')) ORDER BY distance LIMIT 3;",
                         tbl, query);
                ASSERT(exec_scan_sql(db, sql, scan_cb, &streamed) == SQLITE_OK, "TurboQuant streaming scan executes");
                ASSERT(streamed.count == 3, "TurboQuant streaming scan returns rows");
            }
        }
    }

    snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s', 'v', 'qtype=TURBO2');", tbl);
    ASSERT(exec_sql(db, sql) == SQLITE_OK, "vector_quantize qtype=TURBO2 executes");

    snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s', 'v', 'qtype=TURBO,qbits=5');", tbl);
    {
        char *err = NULL;
        int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
        ASSERT(rc != SQLITE_OK, "TurboQuant rejects unsupported qbits");
        sqlite3_free(err);
    }

    {
        const char *wr = "tq_without_rowid";
        snprintf(sql, sizeof(sql), "CREATE TABLE \"%s\" (id INTEGER PRIMARY KEY, v BLOB) WITHOUT ROWID;", wr);
        ASSERT(exec_sql(db, sql) == SQLITE_OK, "TurboQuant WITHOUT ROWID table creates");
        for (int i = 0; i < 8; ++i) {
            snprintf(sql, sizeof(sql), "INSERT INTO \"%s\" (id, v) VALUES (%d, vector_as_f32('%s'));", wr, i + 1, vecs[i]);
            ASSERT(exec_sql(db, sql) == SQLITE_OK, "TurboQuant WITHOUT ROWID insert");
        }
        snprintf(sql, sizeof(sql), "SELECT vector_init('%s', 'v', 'type=f32,dimension=%d,distance=DOT');", wr, dim);
        ASSERT(exec_sql(db, sql) == SQLITE_OK, "TurboQuant WITHOUT ROWID vector_init");
        snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s', 'v', 'qtype=TURBO,qbits=2');", wr);
        ASSERT(exec_sql(db, sql) == SQLITE_OK, "TurboQuant WITHOUT ROWID quantize");
        scan_result wr_result = {0};
        snprintf(sql, sizeof(sql), "SELECT id, distance FROM vector_quantize_scan('%s', 'v', vector_as_f32('%s'), 1);", wr, query);
        ASSERT(sqlite3_exec(db, sql, scan_cb, &wr_result, NULL) == SQLITE_OK, "TurboQuant WITHOUT ROWID scan");
        ASSERT(wr_result.count == 1 && wr_result.ids[0] == 1, "TurboQuant WITHOUT ROWID top-1");
    }

    {
        const char *corrupt = "tq_corrupt";
        if (setup_table(db, corrupt, "f32", "DOT", dim, vecs, 8) == 0) {
            snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s', 'v', 'qtype=TURBO,qbits=4,max_memory=0');", corrupt);
            ASSERT(exec_sql(db, sql) == SQLITE_OK, "TurboQuant corrupt-fixture quantize");
            snprintf(sql, sizeof(sql), "UPDATE \"vector0_%s_v\" SET data = substr(data, 1, 3) WHERE rowid = (SELECT rowid FROM \"vector0_%s_v\" LIMIT 1);", corrupt, corrupt);
            ASSERT(exec_sql(db, sql) == SQLITE_OK, "TurboQuant corrupt-fixture truncates blob");
            snprintf(sql, sizeof(sql), "SELECT id, distance FROM vector_quantize_scan('%s', 'v', vector_as_f32('%s'), 1);", corrupt, query);
            char *err = NULL;
            int rc = sqlite3_exec(db, sql, NULL, NULL, &err);
            ASSERT(rc != SQLITE_OK, "TurboQuant rejects undersized quant blob");
            sqlite3_free(err);
        } else {
            ASSERT(0, "TurboQuant corrupt-fixture setup");
        }
    }

    ASSERT(open_stmt_count(db) == 0, "no open statements after TurboQuant test");
}

static void test_turboquant_edge_dimensions(sqlite3 *db) {
    printf("\n=== TurboQuant Edge Dimensions ===\n");

    sqlite3 *edge_db = NULL;
    char *errmsg = NULL;
    int rc = sqlite3_open(":memory:", &edge_db);
    ASSERT(rc == SQLITE_OK, "TurboQuant edge-dimension database opens");
    if (rc != SQLITE_OK) return;
    ASSERT(sqlite3_vector_init(edge_db, &errmsg, NULL) == SQLITE_OK, "TurboQuant edge-dimension extension init");
    sqlite3_free(errmsg);
    db = edge_db;

    const int dims[] = {1, 3, 7, 15, 17, 769};
    const int bits_values[] = {2, 3, 4};
    const char *distances[] = {"DOT", "COSINE", "L2", "SQUARED_L2"};
    const int nrows = 4;
    char sql[512];
    char msg[192];

    ASSERT(open_stmt_count(db) == 0, "no open statements before TurboQuant edge-dimension test");

    for (int di = 0; di < (int)(sizeof(dims) / sizeof(dims[0])); ++di) {
        int dim = dims[di];
        float *rows = (float *)sqlite3_malloc64((sqlite3_uint64)nrows * (sqlite3_uint64)dim * sizeof(float));
        if (!rows) {
            ASSERT(0, "TurboQuant edge-dimension fixture allocates rows");
            sqlite3_close(edge_db);
            return;
        }

        for (int j = 0; j < dim; ++j) {
            rows[j] = 1.0f;
            rows[dim + j] = -1.0f;
            rows[2 * dim + j] = 0.0f;
            rows[3 * dim + j] = 0.0f;
        }
        rows[3 * dim + (dim > 1 ? dim - 1 : 0)] = dim > 1 ? 1.0f : -0.5f;

        for (int d = 0; d < 4; ++d) {
            for (int b = 0; b < 3; ++b) {
                int bits = bits_values[b];
                char tname[80];
                snprintf(tname, sizeof(tname), "tq_edge_%s_%d_%d", distances[d], dim, bits);

                if (setup_f32_blob_table(db, tname, distances[d], dim, rows, nrows) != 0) {
                    snprintf(msg, sizeof(msg), "TurboQuant edge setup %s dim=%d qbits=%d", distances[d], dim, bits);
                    ASSERT(0, msg);
                    sqlite3_free(rows);
                    sqlite3_close(edge_db);
                    return;
                }

                snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s', 'v', 'qtype=TURBO,qbits=%d,max_memory=0');", tname, bits);
                snprintf(msg, sizeof(msg), "TurboQuant edge quantize %s dim=%d qbits=%d", distances[d], dim, bits);
                ASSERT(exec_sql(db, sql) == SQLITE_OK, msg);

                scan_result exact = {0};
                snprintf(sql, sizeof(sql), "SELECT id, distance FROM vector_full_scan('%s', 'v', ?1, 1);", tname);
                snprintf(msg, sizeof(msg), "TurboQuant edge exact scan %s dim=%d qbits=%d", distances[d], dim, bits);
                ASSERT(exec_bound_scan_sql(db, sql, rows, dim, &exact) == SQLITE_OK, msg);
                snprintf(msg, sizeof(msg), "TurboQuant edge exact returns one row %s dim=%d qbits=%d", distances[d], dim, bits);
                ASSERT(exact.count == 1, msg);

                scan_result approx = {0};
                snprintf(sql, sizeof(sql), "SELECT id, distance FROM vector_quantize_scan('%s', 'v', ?1, 1);", tname);
                snprintf(msg, sizeof(msg), "TurboQuant edge scan %s dim=%d qbits=%d", distances[d], dim, bits);
                ASSERT(exec_bound_scan_sql(db, sql, rows, dim, &approx) == SQLITE_OK, msg);

                snprintf(msg, sizeof(msg), "TurboQuant edge scan returns one row %s dim=%d qbits=%d", distances[d], dim, bits);
                ASSERT(approx.count == 1 && approx.ids[0] >= 1 && approx.ids[0] <= nrows, msg);

                scan_result streamed = {0};
                snprintf(sql, sizeof(sql), "SELECT id, distance FROM vector_quantize_scan('%s', 'v', ?1) LIMIT 2;", tname);
                snprintf(msg, sizeof(msg), "TurboQuant edge streaming scan %s dim=%d qbits=%d", distances[d], dim, bits);
                ASSERT(exec_bound_scan_sql(db, sql, rows, dim, &streamed) == SQLITE_OK, msg);
                snprintf(msg, sizeof(msg), "TurboQuant edge streaming returns rows %s dim=%d qbits=%d", distances[d], dim, bits);
                ASSERT(streamed.count == 2, msg);
            }
        }

        sqlite3_free(rows);
    }

    ASSERT(open_stmt_count(db) == 0, "no open statements after TurboQuant edge-dimension test");
    sqlite3_close(edge_db);
}

static void test_turboquant_reopen(void) {
    printf("\n=== TurboQuant Reopen ===\n");

    char path[256];
    snprintf(path, sizeof(path), "/tmp/sqlite_vector_turboquant_%ld.db", (long)getpid());
    unlink(path);

    const char *vecs[] = {
        "[1, 1, 1, 1, 0, 0, 0, 0]",
        "[0, 0, 0, 0, 1, 1, 1, 1]",
        "[-1, -1, -1, -1, 0, 0, 0, 0]"
    };
    const char *query = "[1, 1, 1, 1, 0, 0, 0, 0]";
    const char *tbl = "tq_reopen";
    char sql[1024];
    sqlite3 *db = NULL;
    char *errmsg = NULL;

    int rc = sqlite3_open(path, &db);
    ASSERT(rc == SQLITE_OK, "TurboQuant reopen database opens");
    ASSERT(sqlite3_vector_init(db, &errmsg, NULL) == SQLITE_OK, "TurboQuant reopen extension init");
    if (setup_table(db, tbl, "f32", "DOT", 8, vecs, 3) == 0) {
        snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s', 'v', 'qtype=TURBO,qbits=2,max_memory=0');", tbl);
        ASSERT(exec_sql(db, sql) == SQLITE_OK, "TurboQuant reopen quantize");
    } else {
        ASSERT(0, "TurboQuant reopen setup");
    }
    ASSERT(open_stmt_count(db) == 0, "TurboQuant reopen no open statements before close");
    sqlite3_close(db);
    db = NULL;

    rc = sqlite3_open(path, &db);
    ASSERT(rc == SQLITE_OK, "TurboQuant reopen database reopens");
    errmsg = NULL;
    ASSERT(sqlite3_vector_init(db, &errmsg, NULL) == SQLITE_OK, "TurboQuant reopen extension re-init");
    snprintf(sql, sizeof(sql), "SELECT vector_init('%s', 'v', 'type=f32,dimension=8,distance=DOT');", tbl);
    ASSERT(exec_sql(db, sql) == SQLITE_OK, "TurboQuant reopen vector_init reloads metadata");

    scan_result r = {0};
    snprintf(sql, sizeof(sql), "SELECT id, distance FROM vector_quantize_scan('%s', 'v', vector_as_f32('%s'), 1);", tbl, query);
    ASSERT(sqlite3_exec(db, sql, scan_cb, &r, NULL) == SQLITE_OK, "TurboQuant reopen scan executes");
    ASSERT(r.count == 1 && r.ids[0] == 1, "TurboQuant reopen top-1");
    ASSERT(open_stmt_count(db) == 0, "TurboQuant reopen no open statements after scan");

    sqlite3_close(db);
    unlink(path);
}

/* ---------- Test vectors ---------- */

/* 4-dimensional float vectors for numeric types */
static const char *float_vecs[] = {
    "[1.0, 0.0, 0.0, 0.0]",
    "[0.0, 1.0, 0.0, 0.0]",
    "[0.0, 0.0, 1.0, 0.0]",
    "[0.0, 0.0, 0.0, 1.0]",
    "[1.0, 1.0, 0.0, 0.0]",
    "[0.0, 1.0, 1.0, 0.0]",
    "[0.0, 0.0, 1.0, 1.0]",
    "[1.0, 1.0, 1.0, 0.0]",
    "[0.0, 1.0, 1.0, 1.0]",
    "[1.0, 1.0, 1.0, 1.0]",
};
static const int float_nvecs = 10;
static const char *float_query = "[0.5, 0.5, 0.5, 0.5]";

/* Integer vectors (0-255 range for U8, -128..127 for I8) */
static const char *int_vecs[] = {
    "[10, 0, 0, 0]",
    "[0, 10, 0, 0]",
    "[0, 0, 10, 0]",
    "[0, 0, 0, 10]",
    "[10, 10, 0, 0]",
    "[0, 10, 10, 0]",
    "[0, 0, 10, 10]",
    "[10, 10, 10, 0]",
    "[0, 10, 10, 10]",
    "[10, 10, 10, 10]",
};
static const int int_nvecs = 10;
static const char *int_query = "[5, 5, 5, 5]";

/* 8-dimensional BIT vectors (0 or 1 values) */
static const char *bit_vecs[] = {
    "[1, 0, 0, 0, 0, 0, 0, 0]",
    "[0, 1, 0, 0, 0, 0, 0, 0]",
    "[0, 0, 1, 0, 0, 0, 0, 0]",
    "[0, 0, 0, 1, 0, 0, 0, 0]",
    "[1, 1, 0, 0, 0, 0, 0, 0]",
    "[0, 1, 1, 0, 0, 0, 0, 0]",
    "[0, 0, 1, 1, 0, 0, 0, 0]",
    "[1, 1, 1, 0, 0, 0, 0, 0]",
    "[0, 1, 1, 1, 0, 0, 0, 0]",
    "[1, 1, 1, 1, 0, 0, 0, 0]",
};
static const int bit_nvecs = 10;
static const char *bit_query = "[1, 0, 1, 0, 1, 0, 1, 0]";

/* ---------- Test: distance function values ---------- */

typedef struct {
    const char *distance_name;
    double eps_f32;
    double eps_f16;
    double eps_bf16;
    double expected[10];
} expected_distance_case;

static const char *distance_vecs[] = {
    "[1.0, 2.0, 0.0, -1.0]",
    "[0.5, -1.5, 2.0, 1.0]",
    "[-2.0, 0.0, 1.0, 0.5]",
    "[3.0, 1.0, -1.0, 2.0]",
    "[-0.5, 2.5, 1.5, -2.0]",
    "[1.5, 1.5, 1.5, 1.5]",
    "[-1.0, -2.0, 0.5, 3.0]",
    "[2.0, -0.5, -2.5, 0.0]",
    "[0.0, 3.0, -1.0, -1.5]",
    "[-1.5, 0.5, 2.5, -0.5]"
};
static const int distance_nvecs = 10;
static const char *distance_query = "[0.75, -0.25, 1.25, -0.75]";
static const char *distance_int_vecs[] = {
    "[10, 2, 0, 7]",
    "[3, 14, 9, 1]",
    "[20, 5, 4, 12]",
    "[8, 8, 8, 8]",
    "[1, 0, 15, 6]",
    "[12, 18, 2, 4]",
    "[6, 3, 11, 19]",
    "[16, 7, 13, 5]",
    "[4, 20, 1, 10]",
    "[9, 11, 6, 14]"
};
static const int distance_int_nvecs = 10;
static const char *distance_int_query = "[7, 9, 5, 11]";

static double eps_for_type(const expected_distance_case *tc, const char *vtype) {
    if (strcasecmp(vtype, "f16") == 0) return tc->eps_f16;
    if (strcasecmp(vtype, "bf16") == 0) return tc->eps_bf16;
    return tc->eps_f32;
}

static void test_one_distance_case(sqlite3 *db, const char *vtype, const expected_distance_case *tc) {
    char tbl[64];
    char sql[1024];
    char msg[256];
    double eps = eps_for_type(tc, vtype);

    snprintf(tbl, sizeof(tbl), "tdist_%s_%s", tc->distance_name, vtype);
    for (char *p = tbl; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;

    if (setup_table(db, tbl, vtype, tc->distance_name, 4, distance_vecs, distance_nvecs) != 0) {
        snprintf(msg, sizeof(msg), "%s/%s distance setup", vtype, tc->distance_name);
        ASSERT(0, msg);
        return;
    }

    scan_result r = {0};
    snprintf(sql, sizeof(sql),
             "SELECT id, distance FROM vector_full_scan('%s', 'v', vector_as_%s('%s')) ORDER BY id;",
             tbl, vtype, distance_query);
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
    snprintf(msg, sizeof(msg), "%s/%s distance query executes", vtype, tc->distance_name);
    ASSERT(rc == SQLITE_OK, msg);
    if (err) { printf("  err: %s\n", err); sqlite3_free(err); }
    if (rc != SQLITE_OK) return;

    snprintf(msg, sizeof(msg), "%s/%s distance query returns all rows", vtype, tc->distance_name);
    ASSERT(r.count == distance_nvecs, msg);
    if (r.count != distance_nvecs) return;

    for (int i = 0; i < distance_nvecs; i++) {
        int id_ok = (r.ids[i] == (i + 1));
        snprintf(msg, sizeof(msg), "%s/%s row id matches expected (row %d)",
                 vtype, tc->distance_name, i + 1);
        ASSERT(id_ok, msg);

        double diff = fabs(r.distances[i] - tc->expected[i]);
        int within_eps = diff <= eps;
        snprintf(msg, sizeof(msg), "%s/%s distance within epsilon (id=%d, diff=%.8g, eps=%.3g)",
                 vtype, tc->distance_name, i + 1, diff, eps);
        ASSERT(within_eps, msg);
    }
}

static void test_distance_functions_float(sqlite3 *db) {
    const expected_distance_case cases[] = {
        {
            .distance_name = "L2",
            .eps_f32 = 1e-6,
            .eps_f16 = 1e-2,
            .eps_bf16 = 5e-2,
            .expected = {
                2.598076211353316,
                2.291287847477920,
                3.041381265149110,
                4.387482193696061,
                3.278719262151000,
                2.958039891549808,
                4.555216789572150,
                4.031128874149275,
                4.092676385936225,
                2.692582403567252
            }
        },
        {
            .distance_name = "SQUARED_L2",
            .eps_f32 = 1e-6,
            .eps_f16 = 5e-2,
            .eps_bf16 = 2e-1,
            .expected = {6.75, 5.25, 9.25, 19.25, 10.75, 8.75, 20.75, 16.25, 16.75, 7.25}
        },
        {
            .distance_name = "COSINE",
            .eps_f32 = 1e-5,
            .eps_f16 = 1e-2,
            .eps_bf16 = 5e-2,
            .expected = {
                0.753817018041334,
                0.449518117436820,
                1.164487923739942,
                1.116774841624228,
                0.598909685625288,
                0.698488655422236,
                1.299521148936577,
                1.279145263119541,
                1.150755672288882,
                0.547732983133355
            }
        },
        {
            .distance_name = "DOT",
            .eps_f32 = 1e-6,
            .eps_f16 = 1e-2,
            .eps_bf16 = 5e-2,
            .expected = {-1.0, -2.5, 0.625, 0.75, -2.375, -1.5, 1.875, 1.5, 0.875, -2.25}
        },
        {
            .distance_name = "L1",
            .eps_f32 = 1e-6,
            .eps_f16 = 1e-2,
            .eps_bf16 = 5e-2,
            .expected = {4.0, 4.0, 4.5, 8.5, 5.5, 5.0, 8.0, 6.0, 7.0, 4.5}
        }
    };
    const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));
    const char *types[] = {"f32", "f16", "bf16"};

    for (int t = 0; t < 3; t++) {
        for (int i = 0; i < ncases; i++) {
            test_one_distance_case(db, types[t], &cases[i]);
        }
    }
}

typedef struct {
    const char *distance_name;
    double eps_i8;
    double eps_u8;
    double expected[10];
} expected_int_distance_case;

static double eps_for_int_type(const expected_int_distance_case *tc, const char *vtype) {
    if (strcasecmp(vtype, "i8") == 0) return tc->eps_i8;
    return tc->eps_u8;
}

static void test_one_int_distance_case(sqlite3 *db, const char *vtype, const expected_int_distance_case *tc) {
    char tbl[64];
    char sql[1024];
    char msg[256];
    double eps = eps_for_int_type(tc, vtype);

    snprintf(tbl, sizeof(tbl), "tdist_%s_%s", tc->distance_name, vtype);
    for (char *p = tbl; *p; p++) if (*p >= 'A' && *p <= 'Z') *p += 32;

    if (setup_table(db, tbl, vtype, tc->distance_name, 4, distance_int_vecs, distance_int_nvecs) != 0) {
        snprintf(msg, sizeof(msg), "%s/%s int distance setup", vtype, tc->distance_name);
        ASSERT(0, msg);
        return;
    }

    scan_result r = {0};
    snprintf(sql, sizeof(sql),
             "SELECT id, distance FROM vector_full_scan('%s', 'v', vector_as_%s('%s')) ORDER BY id;",
             tbl, vtype, distance_int_query);
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
    snprintf(msg, sizeof(msg), "%s/%s int distance query executes", vtype, tc->distance_name);
    ASSERT(rc == SQLITE_OK, msg);
    if (err) { printf("  err: %s\n", err); sqlite3_free(err); }
    if (rc != SQLITE_OK) return;

    snprintf(msg, sizeof(msg), "%s/%s int distance query returns all rows", vtype, tc->distance_name);
    ASSERT(r.count == distance_int_nvecs, msg);
    if (r.count != distance_int_nvecs) return;

    for (int i = 0; i < distance_int_nvecs; i++) {
        int id_ok = (r.ids[i] == (i + 1));
        snprintf(msg, sizeof(msg), "%s/%s int row id matches expected (row %d)",
                 vtype, tc->distance_name, i + 1);
        ASSERT(id_ok, msg);

        double diff = fabs(r.distances[i] - tc->expected[i]);
        int within_eps = diff <= eps;
        snprintf(msg, sizeof(msg), "%s/%s int distance within epsilon (id=%d, diff=%.8g, eps=%.3g)",
                 vtype, tc->distance_name, i + 1, diff, eps);
        ASSERT(within_eps, msg);
    }
}

static void test_distance_functions_int(sqlite3 *db) {
    const expected_int_distance_case cases[] = {
        {
            .distance_name = "L2",
            .eps_i8 = 1e-6,
            .eps_u8 = 1e-6,
            .expected = {
                9.949874371066199,
                12.529964086141668,
                13.674794331177344,
                4.472135954999580,
                15.556349186104045,
                12.806248474865697,
                11.704699910719626,
                13.601470508735444,
                12.124355652982141,
                4.242640687119285
            }
        },
        {
            .distance_name = "SQUARED_L2",
            .eps_i8 = 1e-6,
            .eps_u8 = 1e-6,
            .expected = {99.0, 157.0, 187.0, 20.0, 242.0, 164.0, 137.0, 185.0, 147.0, 18.0}
        },
        {
            .distance_name = "COSINE",
            .eps_i8 = 1e-6,
            .eps_u8 = 1e-6,
            .expected = {
                0.197058901598547,
                0.278725549597720,
                0.161317797973194,
                0.036913175313846,
                0.449627749704491,
                0.182558273343614,
                0.126858993881120,
                0.205091387999948,
                0.144927951966812,
                0.000283884548207
            }
        },
        {
            .distance_name = "DOT",
            .eps_i8 = 1e-6,
            .eps_u8 = 1e-6,
            .expected = {-165.0, -203.0, -337.0, -256.0, -148.0, -300.0, -333.0, -295.0, -323.0, -346.0}
        },
        {
            .distance_name = "L1",
            .eps_i8 = 1e-6,
            .eps_u8 = 1e-6,
            .expected = {19.0, 23.0, 19.0, 8.0, 30.0, 24.0, 21.0, 25.0, 19.0, 8.0}
        }
    };
    const int ncases = (int)(sizeof(cases) / sizeof(cases[0]));
    const char *types[] = {"i8", "u8"};

    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < ncases; i++) {
            test_one_int_distance_case(db, types[t], &cases[i]);
        }
    }
}

static void test_distance_functions_hamming(sqlite3 *db) {
    const char *tbl = "tdist_hamming_bit";
    const double expected[] = {3.0, 5.0, 3.0, 5.0, 4.0, 4.0, 4.0, 3.0, 5.0, 4.0};
    char sql[1024];
    char msg[256];

    if (setup_table(db, tbl, "bit", "HAMMING", 8, bit_vecs, bit_nvecs) != 0) {
        ASSERT(0, "bit/HAMMING distance setup");
        return;
    }

    scan_result r = {0};
    snprintf(sql, sizeof(sql),
             "SELECT id, distance FROM vector_full_scan('%s', 'v', vector_as_bit('%s')) ORDER BY id;",
             tbl, bit_query);
    char *err = NULL;
    int rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
    ASSERT(rc == SQLITE_OK, "bit/HAMMING distance query executes");
    if (err) { printf("  err: %s\n", err); sqlite3_free(err); }
    if (rc != SQLITE_OK) return;

    ASSERT(r.count == bit_nvecs, "bit/HAMMING distance query returns all rows");
    if (r.count != bit_nvecs) return;

    for (int i = 0; i < bit_nvecs; i++) {
        int id_ok = (r.ids[i] == (i + 1));
        snprintf(msg, sizeof(msg), "bit/HAMMING row id matches expected (row %d)", i + 1);
        ASSERT(id_ok, msg);

        int exact_match = (r.distances[i] == expected[i]);
        snprintf(msg, sizeof(msg), "bit/HAMMING distance matches exactly (id=%d)", i + 1);
        ASSERT(exact_match, msg);
    }
}

/* ---------- Test: malformed input and failure paths ---------- */

/* Like exec_scan_sql but keeps the step() return code, which the tests below need in
   order to tell "rejected cleanly" from "produced a row it should not have". */
static int run_scan(sqlite3 *db, const char *sql, scan_result *r) {
    sqlite3_stmt *stmt = NULL;
    int rc = sqlite3_prepare_v2(db, sql, -1, &stmt, NULL);
    if (rc != SQLITE_OK) return rc;

    while ((rc = sqlite3_step(stmt)) == SQLITE_ROW) {
        if (r->count < 64) {
            r->ids[r->count] = sqlite3_column_int(stmt, 0);
            r->distances[r->count] = sqlite3_column_double(stmt, 1);
        }
        r->count++;
    }

    sqlite3_finalize(stmt);
    return (rc == SQLITE_DONE) ? SQLITE_OK : rc;
}

static void test_malformed_input(sqlite3 *db) {
    printf("\n=== Malformed input and failure paths ===\n");
    char sql[1024];

    /*
     * A k whose byte count overflows an int. sqlite3_malloc() takes an int, so
     * k=536870913 asked for 0x100000008 bytes, got 8, and the memset that follows
     * overran the top-k arrays. It has to fail cleanly, and leave a usable cursor.
     */
    {
        const char *tbl = "tbad_bigk";
        if (setup_table(db, tbl, "f32", "L2", 4, float_vecs, float_nvecs) != 0) {
            ASSERT(0, "huge k setup");
        } else {
            scan_result r = {0};
            snprintf(sql, sizeof(sql),
                     "SELECT id, distance FROM vector_full_scan('%s','v',vector_as_f32('%s'),536870913);",
                     tbl, float_query);
            int rc = run_scan(db, sql, &r);
            ASSERT(rc != SQLITE_OK, "huge k is rejected instead of overflowing the top-k arrays");

            scan_result r2 = {0};
            snprintf(sql, sizeof(sql),
                     "SELECT id, distance FROM vector_full_scan('%s','v',vector_as_f32('%s'),3);",
                     tbl, float_query);
            rc = run_scan(db, sql, &r2);
            ASSERT(rc == SQLITE_OK && r2.count == 3, "a sane k still works after a huge one");
        }
    }

    /*
     * _sqliteai_vector is an ordinary writable table and PRIMARY KEY on a rowid table
     * does not imply NOT NULL, so a row with a NULL key used to reach strcmp(NULL).
     */
    {
        const char *tbl = "tbad_nullkey";
        snprintf(sql, sizeof(sql), "CREATE TABLE \"%s\" (id INTEGER PRIMARY KEY, v BLOB);", tbl);
        exec_sql(db, sql);
        snprintf(sql, sizeof(sql),
                 "INSERT INTO _sqliteai_vector (tblname,colname,key,value) VALUES ('%s','v',NULL,1);", tbl);
        int rc = exec_sql(db, sql);
        ASSERT(rc == SQLITE_OK, "a NULL metadata key can be stored");

        snprintf(sql, sizeof(sql),
                 "SELECT vector_init('%s','v','type=f32,dimension=4,distance=L2');", tbl);
        rc = exec_sql(db, sql);
        ASSERT(rc == SQLITE_OK, "vector_init skips a NULL metadata key instead of crashing");
    }

    /*
     * A quant chunk that claims zero vectors. Zero is the streaming scans' "load the
     * next chunk" sentinel, so such a chunk slipped past the length check and one row
     * was read from past the end of its blob.
     */
    {
        const char *tbl = "tbad_zerocount";
        if (setup_table(db, tbl, "f32", "L2", 4, float_vecs, float_nvecs) != 0) {
            ASSERT(0, "zero-count chunk setup");
        } else {
            snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s','v','qtype=UINT8');", tbl);
            ASSERT(exec_sql(db, sql) == SQLITE_OK, "zero-count chunk: quantize succeeds");

            scan_result before = {0};
            snprintf(sql, sizeof(sql),
                     "SELECT id, distance FROM vector_quantize_scan('%s','v',vector_as_f32('%s'));",
                     tbl, float_query);
            ASSERT(run_scan(db, sql, &before) == SQLITE_OK, "zero-count chunk: scan works before tampering");

            snprintf(sql, sizeof(sql),
                     "INSERT INTO \"vector0_%s_v\" (rowid1,rowid2,counter,data) VALUES (0,0,0,x'00');", tbl);
            ASSERT(exec_sql(db, sql) == SQLITE_OK, "zero-count chunk can be stored");

            scan_result streaming = {0};
            snprintf(sql, sizeof(sql),
                     "SELECT id, distance FROM vector_quantize_scan('%s','v',vector_as_f32('%s'));",
                     tbl, float_query);
            int rc = run_scan(db, sql, &streaming);
            ASSERT(rc != SQLITE_OK, "streaming scan rejects a chunk claiming zero vectors");
            ASSERT(streaming.count <= before.count, "streaming scan invents no extra row");

            scan_result topk = {0};
            snprintf(sql, sizeof(sql),
                     "SELECT id, distance FROM vector_quantize_scan('%s','v',vector_as_f32('%s'),5);",
                     tbl, float_query);
            rc = run_scan(db, sql, &topk);
            ASSERT(rc != SQLITE_OK, "top-k scan rejects a chunk claiming zero vectors");
        }
    }

    /*
     * A vector_quantize that fails part way through. The savepoint rolls the shadow
     * table back, so the in-memory q_type/scale/offset have to go back too: otherwise
     * the next scan reads the chunks at the wrong stride and reports made-up rowids.
     * 1BIT skips the min/max pass, so the undersized blob is caught in the later one.
     */
    {
        const char *tbl = "tbad_rollback";
        if (setup_table(db, tbl, "f32", "L2", 4, float_vecs, float_nvecs) != 0) {
            ASSERT(0, "quantize rollback setup");
        } else {
            snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s','v','qtype=UINT8');", tbl);
            ASSERT(exec_sql(db, sql) == SQLITE_OK, "quantize rollback: UINT8 quantize succeeds");

            scan_result before = {0};
            snprintf(sql, sizeof(sql),
                     "SELECT id, distance FROM vector_quantize_scan('%s','v',vector_as_f32('%s'),5) ORDER BY distance;",
                     tbl, float_query);
            ASSERT(run_scan(db, sql, &before) == SQLITE_OK, "quantize rollback: scan works before");

            snprintf(sql, sizeof(sql), "INSERT INTO \"%s\" (id, v) VALUES (9999, x'0001');", tbl);
            ASSERT(exec_sql(db, sql) == SQLITE_OK, "quantize rollback: undersized blob stored");

            snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s','v','qtype=1BIT');", tbl);
            ASSERT(exec_sql(db, sql) != SQLITE_OK, "quantize rollback: 1BIT quantize fails on the bad blob");

            snprintf(sql, sizeof(sql), "DELETE FROM \"%s\" WHERE id = 9999;", tbl);
            exec_sql(db, sql);

            scan_result after = {0};
            snprintf(sql, sizeof(sql),
                     "SELECT id, distance FROM vector_quantize_scan('%s','v',vector_as_f32('%s'),5) ORDER BY distance;",
                     tbl, float_query);
            ASSERT(run_scan(db, sql, &after) == SQLITE_OK, "quantize rollback: scan still works after the failure");
            ASSERT(after.count == before.count, "quantize rollback: same number of rows as before");

            int same = (after.count == before.count);
            for (int i = 0; same && i < after.count; i++) {
                if (after.ids[i] != before.ids[i]) same = 0;
                if (after.distances[i] != before.distances[i]) same = 0;
            }
            ASSERT(same, "quantize rollback: same rowids and distances as before the failed call");
        }
    }

    /*
     * The optional dimension argument has to hold a BLOB to the size it names, as it
     * already did for TEXT and for BIT: a wrong-sized vector is otherwise stored and
     * then skipped in silence by every scan.
     */
    {
        sqlite3_stmt *stmt = NULL;
        int rc = sqlite3_prepare_v2(db, "SELECT length(vector_as_f32(x'0000000000000000', 4));", -1, &stmt, NULL);
        if (rc == SQLITE_OK) rc = sqlite3_step(stmt);
        ASSERT(rc != SQLITE_ROW, "vector_as_f32 rejects a BLOB that does not match the given dimension");
        sqlite3_finalize(stmt);

        stmt = NULL;
        rc = sqlite3_prepare_v2(db, "SELECT length(vector_as_f32(x'0000000000000000', 2));", -1, &stmt, NULL);
        if (rc == SQLITE_OK) rc = sqlite3_step(stmt);
        ASSERT(rc == SQLITE_ROW && sqlite3_column_int(stmt, 0) == 8, "vector_as_f32 accepts a BLOB of the right dimension");
        sqlite3_finalize(stmt);

        /*
         * The size in the rejection message is a size_t behind a %lld, so it has to be
         * reported whole. context_result_error() carries no printf format attribute, so
         * nothing but a runtime check covers the conversion.
         */
        stmt = NULL;
        rc = sqlite3_prepare_v2(db, "SELECT vector_as_f32(x'00000000', 16777216);", -1, &stmt, NULL);
        if (rc == SQLITE_OK) rc = sqlite3_step(stmt);
        ASSERT(rc != SQLITE_ROW, "vector_as_f32 rejects a 4-byte BLOB offered as 16777216 dimensions");
        {
            const char *msg = sqlite3_errmsg(db);
            ASSERT(msg && strstr(msg, "67108864") != NULL, "the rejection reports the expected size");
            if (!msg || !strstr(msg, "67108864")) printf("  msg: %s\n", msg ? msg : "(none)");
        }
        sqlite3_finalize(stmt);

        /* past the dimension bound it is refused before any size is computed at all */
        stmt = NULL;
        rc = sqlite3_prepare_v2(db, "SELECT vector_as_f32(x'00000000', 1073741824);", -1, &stmt, NULL);
        if (rc == SQLITE_OK) rc = sqlite3_step(stmt);
        ASSERT(rc != SQLITE_ROW, "vector_as_f32 rejects a dimension past the bound");
        sqlite3_finalize(stmt);

        stmt = NULL;
        rc = sqlite3_prepare_v2(db, "SELECT length(vector_as_f32(x'0000000000000000'));", -1, &stmt, NULL);
        if (rc == SQLITE_OK) rc = sqlite3_step(stmt);
        ASSERT(rc == SQLITE_ROW && sqlite3_column_int(stmt, 0) == 8, "vector_as_f32 still accepts a BLOB with no dimension given");
        sqlite3_finalize(stmt);
    }

    /*
     * An unbounded dimension overflowed the size arithmetic downstream. The one that
     * crashed outright: (dim/2)*12 in turbo_rotation_plan_init() wraps negative past
     * 357913941, which skipped the allocation of the coordinate arrays and still
     * reported the plan as valid, so building a TurboQuant index on an EMPTY table
     * wrote through a NULL pointer. The dimension is now range-checked where it is
     * parsed, which is upstream of every one of those computations.
     */
    {
        const char *tbl = "tbad_dim";
        snprintf(sql, sizeof(sql), "CREATE TABLE \"%s\" (id INTEGER PRIMARY KEY, v BLOB);", tbl);
        exec_sql(db, sql);

        struct { const char *dim; int ok; const char *what; } cases[] = {
            {"4",          1, "a small dimension is accepted"},
            {"16777216",   1, "the largest allowed dimension is accepted"},
            {"16777217",   0, "one past the bound is rejected"},
            {"400000000",  0, "the dimension that overflowed the TurboQuant plan is rejected"},
            {"2147483647", 0, "INT_MAX is rejected"},
            /* (int)strtol() used to wrap this to 4 and accept it as a valid dimension */
            {"4294967300", 0, "a value above INT_MAX is rejected, not wrapped"},
        };

        for (unsigned i = 0; i < sizeof(cases) / sizeof(*cases); i++) {
            char msg[160];
            /* a fresh column name each time: vector_init keeps the first dimension it
               is given for a table/column pair and rejects later disagreements */
            snprintf(sql, sizeof(sql), "ALTER TABLE \"%s\" ADD COLUMN c%u BLOB;", tbl, i);
            exec_sql(db, sql);
            snprintf(sql, sizeof(sql), "SELECT vector_init('%s','c%u','type=f32,dimension=%s');", tbl, i, cases[i].dim);
            int rc = exec_sql(db, sql);
            snprintf(msg, sizeof(msg), "dimension=%s: %s", cases[i].dim, cases[i].what);
            ASSERT(cases[i].ok ? (rc == SQLITE_OK) : (rc != SQLITE_OK), msg);
        }

        /* and the quantize that used to crash is now unreachable: init refuses first */
        snprintf(sql, sizeof(sql), "ALTER TABLE \"%s\" ADD COLUMN turbo BLOB;", tbl);
        exec_sql(db, sql);
        snprintf(sql, sizeof(sql), "SELECT vector_init('%s','turbo','type=f32,dimension=400000000,distance=COSINE');", tbl);
        ASSERT(exec_sql(db, sql) != SQLITE_OK, "the TurboQuant crash case is refused at vector_init");
        snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s','turbo','qtype=TURBO');", tbl);
        ASSERT(exec_sql(db, sql) != SQLITE_OK, "and vector_quantize on it reports no context instead of crashing");
    }

    /*
     * max_memory: 0 is a real setting ("use whatever the index needs") and used to be
     * indistinguishable from a parse failure, which was itself silently ignored.
     */
    {
        const char *tbl = "tbad_maxmemory";
        if (setup_table(db, tbl, "f32", "L2", 4, float_vecs, float_nvecs) != 0) {
            ASSERT(0, "max_memory setup");
        } else {
            snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s','v','qtype=UINT8,max_memory=0');", tbl);
            ASSERT(exec_sql(db, sql) == SQLITE_OK, "max_memory=0 is accepted");

            scan_result r = {0};
            snprintf(sql, sizeof(sql),
                     "SELECT id, distance FROM vector_quantize_scan('%s','v',vector_as_f32('%s'),5);",
                     tbl, float_query);
            ASSERT(run_scan(db, sql, &r) == SQLITE_OK && r.count == 5, "max_memory=0 quantizes a usable index");

            snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s','v','qtype=UINT8,max_memory=banana');", tbl);
            ASSERT(exec_sql(db, sql) != SQLITE_OK, "an unparseable max_memory is reported, not ignored");

            snprintf(sql, sizeof(sql), "SELECT vector_quantize('%s','v','qtype=UINT8,max_memory=1MB');", tbl);
            ASSERT(exec_sql(db, sql) == SQLITE_OK, "max_memory with a unit suffix is accepted");
        }
    }
}

/* ---------- Main ---------- */

int main(void) {
    sqlite3 *db;
    int rc = sqlite3_open(":memory:", &db);
    if (rc != SQLITE_OK) {
        printf("FAIL: cannot open :memory: database\n");
        return 1;
    }

    /* Initialize the vector extension */
    char *errmsg = NULL;
    rc = sqlite3_vector_init(db, &errmsg, NULL);
    if (rc != SQLITE_OK) {
        printf("FAIL: sqlite3_vector_init returned %d: %s\n", rc, errmsg ? errmsg : "");
        sqlite3_close(db);
        return 1;
    }

    /* 1. Basics */
    test_basics(db);

    /* 2. vector_full_scan — float types × all distances */
    printf("\n=== vector_full_scan ===\n");
    {
        const char *float_types[] = {"f32", "f16", "bf16"};
        const char *distances[]   = {"L2", "SQUARED_L2", "COSINE", "DOT", "L1"};

        for (int t = 0; t < 3; t++) {
            for (int d = 0; d < 5; d++) {
                test_full_scan(db, float_types[t], distances[d],
                               4, float_vecs, float_nvecs, float_query);
            }
        }

        /* Integer types */
        const char *int_types[] = {"i8", "u8"};
        for (int t = 0; t < 2; t++) {
            for (int d = 0; d < 5; d++) {
                test_full_scan(db, int_types[t], distances[d],
                               4, int_vecs, int_nvecs, int_query);
            }
        }

        /* BIT — only HAMMING */
        test_full_scan(db, "bit", "HAMMING", 8, bit_vecs, bit_nvecs, bit_query);
    }

    /* 3. vector_quantize_scan — all vector types × quantization types */
    printf("\n=== vector_quantize_scan ===\n");
    {
        const char *qtypes[] = {"UINT8", "INT8", "1BIT"};

        /* Float types */
        const char *float_types[] = {"f32", "f16", "bf16"};
        for (int t = 0; t < 3; t++) {
            for (int q = 0; q < 3; q++) {
                test_quantize_scan(db, float_types[t], qtypes[q],
                                   4, float_vecs, float_nvecs, float_query);
            }
        }

        /* Integer types */
        const char *int_types[] = {"i8", "u8"};
        for (int t = 0; t < 2; t++) {
            for (int q = 0; q < 3; q++) {
                test_quantize_scan(db, int_types[t], qtypes[q],
                                   4, int_vecs, int_nvecs, int_query);
            }
        }

        /* BIT — quantize with 1BIT */
        test_quantize_scan(db, "bit", "1BIT", 8, bit_vecs, bit_nvecs, bit_query);
    }

    test_turboquant(db);
    test_turboquant_edge_dimensions(db);
    test_turboquant_reopen();

    /* 4. Streaming ORDER BY (regression test for issue #43) */
    printf("\n=== Streaming ORDER BY ===\n");
    {
        /*
         * Regression test: vector_full_scan_stream with ORDER BY + LIMIT
         * must return results sorted by distance.  A bug in vFullScanBestIndex
         * caused orderByConsumed=1 even in streaming mode, so SQLite skipped
         * sorting and returned rows in table order.
         *
         * We insert vectors in REVERSE distance order (farthest first) so
         * that table-order != distance-order, exposing the bug.
         */
        const char *tbl = "tfs_stream_order";
        const char *vecs_ordered[] = {
            "[10.0, 0.0, 0.0, 0.0]",   /* id 1: far   */
            "[0.0, 10.0, 0.0, 0.0]",   /* id 2: far   */
            "[5.0, 5.0, 0.0, 0.0]",    /* id 3: far   */
            "[2.0, 0.0, 0.0, 0.0]",    /* id 4: mid   */
            "[1.0, 1.0, 0.0, 0.0]",    /* id 5: mid   */
            "[0.6, 0.6, 0.6, 0.6]",    /* id 6: close */
            "[0.4, 0.4, 0.4, 0.4]",    /* id 7: close */
            "[0.5, 0.5, 0.5, 0.5]",    /* id 8: exact */
            "[0.3, 0.3, 0.3, 0.3]",    /* id 9: close */
            "[0.7, 0.7, 0.7, 0.7]",    /* id 10: close */
        };
        const char *qvec = "[0.5, 0.5, 0.5, 0.5]";

        if (setup_table(db, tbl, "f32", "L2", 4, vecs_ordered, 10) == 0) {
            /* Helper callback: distance is column 0 in these queries */
            struct { int count; double distances[64]; } r;

            /* vector_full_scan_stream + JOIN + ORDER BY distance LIMIT */
            {
                memset(&r, 0, sizeof(r));
                char sql[1024];
                snprintf(sql, sizeof(sql),
                    "SELECT vss.distance, t.id"
                    " FROM \"%s\" t"
                    " INNER JOIN vector_full_scan_stream('%s', 'v', vector_as_f32('%s'))"
                    "   AS vss ON vss.rowid = t.id"
                    " ORDER BY vss.distance LIMIT 5;",
                    tbl, tbl, qvec);
                char *err = NULL;
                rc = sqlite3_exec(db, sql, scan_cb_col0, &r, &err);
                ASSERT(rc == SQLITE_OK, "stream+JOIN+ORDER BY executes");
                if (err) { printf("  err: %s\n", err); sqlite3_free(err); }

                ASSERT(r.count == 5, "stream+JOIN+ORDER BY returns 5 rows");

                int sorted = 1;
                for (int i = 1; i < r.count; i++) {
                    if (r.distances[i] < r.distances[i - 1]) sorted = 0;
                }
                ASSERT(sorted, "stream+JOIN+ORDER BY distances sorted");
                ASSERT(r.distances[0] < 0.01, "stream+JOIN+ORDER BY closest is ~0");
            }

            /* vector_full_scan (no k, streaming) + ORDER BY distance LIMIT */
            {
                memset(&r, 0, sizeof(r));
                char sql[1024];
                snprintf(sql, sizeof(sql),
                    "SELECT distance, rowid"
                    " FROM vector_full_scan_stream('%s', 'v', vector_as_f32('%s'))"
                    " ORDER BY distance LIMIT 5;",
                    tbl, qvec);
                char *err = NULL;
                rc = sqlite3_exec(db, sql, scan_cb_col0, &r, &err);
                ASSERT(rc == SQLITE_OK, "stream+ORDER BY executes");
                if (err) { printf("  err: %s\n", err); sqlite3_free(err); }

                ASSERT(r.count == 5, "stream+ORDER BY returns 5 rows");

                int sorted = 1;
                for (int i = 1; i < r.count; i++) {
                    if (r.distances[i] < r.distances[i - 1]) sorted = 0;
                }
                ASSERT(sorted, "stream+ORDER BY distances sorted");
                ASSERT(r.distances[0] < 0.01, "stream+ORDER BY closest is ~0");
            }
        }
    }

    /* 5. Backward-compat aliases */
    printf("\n=== Backward-compat aliases ===\n");
    {
        /* Set up a table for alias tests */
        const char *tbl = "tfs_alias";
        if (setup_table(db, tbl, "f32", "L2", 4, float_vecs, float_nvecs) == 0) {
            /* vector_full_scan_stream */
            {
                scan_result r = {0};
                char sql[512];
                snprintf(sql, sizeof(sql),
                         "SELECT id, distance FROM vector_full_scan_stream('%s', 'v', vector_as_f32('%s')) LIMIT 3;",
                         tbl, float_query);
                char *err = NULL;
                rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
                ASSERT(rc == SQLITE_OK, "vector_full_scan_stream alias works");
                if (err) { printf("  err: %s\n", err); sqlite3_free(err); }
                ASSERT(r.count > 0, "vector_full_scan_stream returns rows");
            }

            /* vector_quantize_scan_stream */
            {
                char sql[512];
                snprintf(sql, sizeof(sql),
                         "SELECT vector_quantize('%s', 'v');", tbl);
                exec_sql(db, sql);

                scan_result r = {0};
                snprintf(sql, sizeof(sql),
                         "SELECT id, distance FROM vector_quantize_scan_stream('%s', 'v', vector_as_f32('%s')) LIMIT 3;",
                         tbl, float_query);
                char *err = NULL;
                rc = sqlite3_exec(db, sql, scan_cb, &r, &err);
                ASSERT(rc == SQLITE_OK, "vector_quantize_scan_stream alias works");
                if (err) { printf("  err: %s\n", err); sqlite3_free(err); }
                ASSERT(r.count > 0, "vector_quantize_scan_stream returns rows");
            }
        }
    }


    /* 5. distance functions */
    printf("\n=== distance_functions ===\n");
    {
        test_distance_functions_float(db);
        test_distance_functions_int(db);
        test_distance_functions_hamming(db);
    }

    /* 6. malformed input and failure paths */
    test_malformed_input(db);


    sqlite3_close(db);

    /* Summary */
    printf("\n========================================\n");
    printf("Results: %d passed, %d failed\n", passes, failures);
    printf("========================================\n");

    return failures > 0 ? 1 : 0;
}

//
//  benchmark.c
//  sqlitevector
//
//  Brute-force k-NN benchmark: one query vector against every row, for each storage and
//  quantization mode the extension supports. Reports throughput and recall against the
//  exact full-precision scan, because speed without recall says nothing.
//
//  Defaults to k=20 over 1,000,000 vectors of dimension 768. Override at build time:
//      make benchmark NVECS=100000 DIM=384 K=10 NQUERIES=20
//
//  The database is in memory, so "on disk" below means the index is read back through
//  SQLite rather than from the extension's preloaded buffer - not filesystem I/O.
//

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "sqlite3.h"

extern int sqlite3_vector_init (sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);

#ifndef NVECS
#define NVECS       1000000
#endif
#ifndef DIM
#define DIM         768
#endif
#ifndef K
#define K           20
#endif
#ifndef NQUERIES
#define NQUERIES    20
#endif
#ifndef DISTANCE
#define DISTANCE    "cosine"
#endif
#ifndef HARDWARE
#define HARDWARE    "<CPU> — <backend> backend"
#endif

// xorshift64*: the data must be identical from run to run and from machine to machine,
// and rand() is neither fast enough nor portable enough for that
static uint64_t rng_state = 0x853c49e6748fea9bULL;
static inline double rng_unit (void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (double)((rng_state * 0x2545F4914F6CDD1DULL) >> 11) / 9007199254740992.0;
}

static double now_seconds (void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void die (sqlite3 *db, const char *what, char *err) {
    fprintf(stderr, "%s: %s\n", what, err ? err : sqlite3_errmsg(db));
    exit(1);
}

static void run_sql (sqlite3 *db, const char *sql) {
    char *err = NULL;
    if (sqlite3_exec(db, sql, NULL, NULL, &err) != SQLITE_OK) die(db, sql, err);
    if (err) sqlite3_free(err);
}

static float queries[NQUERIES][DIM];
static int64_t exact[NQUERIES][K];

// runs every query, optionally scoring the returned rowids against the exact answer
static double measure (sqlite3 *db, const char *tvf, int64_t truth[][K], double *recall_out) {
    char sql[256];
    snprintf(sql, sizeof(sql), "SELECT rowid FROM %s('t','v',?,%d);", tvf, K);

    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, sql, -1, &stmt, NULL) != SQLITE_OK) die(db, sql, NULL);

    double best = 1e30;
    int hits = 0, total = 0;
    for (int q = 0; q < NQUERIES; ++q) {
        int64_t got[K];
        int n = 0;

        double t0 = now_seconds();
        sqlite3_bind_blob(stmt, 1, queries[q], (int)sizeof(queries[q]), SQLITE_STATIC);
        while (sqlite3_step(stmt) == SQLITE_ROW && n < K) got[n++] = sqlite3_column_int64(stmt, 0);
        sqlite3_reset(stmt);
        double elapsed = now_seconds() - t0;
        if (elapsed < best) best = elapsed;

        if (truth) {
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < K; ++j) if (got[i] == truth[q][j]) { ++hits; break; }
            }
            total += K;
        } else {
            memcpy(exact[q], got, sizeof(got));
        }
    }
    sqlite3_finalize(stmt);

    if (recall_out) *recall_out = truth ? (100.0 * hits / total) : 100.0;
    return best;
}

static sqlite3_int64 index_bytes (sqlite3 *db) {
    sqlite3_stmt *stmt = NULL;
    sqlite3_int64 bytes = 0;
    if (sqlite3_prepare_v2(db, "SELECT SUM(LENGTH(data)) FROM vector0_t_v;", -1, &stmt, NULL) == SQLITE_OK) {
        if (sqlite3_step(stmt) == SQLITE_ROW) bytes = sqlite3_column_int64(stmt, 0);
    }
    sqlite3_finalize(stmt);
    return bytes;
}

// 1000000 is hard to read in a table cell
static const char *with_separators (long long n, char *buf, size_t cap) {
    char digits[32];
    snprintf(digits, sizeof(digits), "%lld", n);
    size_t len = strlen(digits), out = 0;
    for (size_t i = 0; i < len && out + 2 < cap; ++i) {
        if (i > 0 && ((len - i) % 3) == 0) buf[out++] = ',';
        buf[out++] = digits[i];
    }
    buf[out] = 0;
    return buf;
}

static void report (const char *label, sqlite3_int64 bytes, double seconds, double recall) {
    printf("| %-24s | %8.1f | %9.2f | %9.1f | %6.1f |\n",
           label,
           bytes ? (double)bytes / (1024.0 * 1024.0) : (double)NVECS * DIM * sizeof(float) / (1024.0 * 1024.0),
           seconds * 1000.0,
           NVECS / seconds / 1e6,
           recall);
}

int main (void) {
    sqlite3 *db = NULL;
    if (sqlite3_open(":memory:", &db) != SQLITE_OK) die(db, "open", NULL);
    if (sqlite3_vector_init(db, NULL, NULL) != SQLITE_OK) die(db, "vector_init", NULL);

    sqlite3_stmt *stmt = NULL;
    sqlite3_prepare_v2(db, "SELECT vector_backend();", -1, &stmt, NULL);
    sqlite3_step(stmt);
    printf("sqlite-vector benchmark - backend %s, SQLite %s\n", sqlite3_column_text(stmt, 0), sqlite3_libversion());
    sqlite3_finalize(stmt);
    printf("%d vectors, dimension %d, %s distance, k=%d, %d queries, best of run\n", NVECS, DIM, DISTANCE, K, NQUERIES);
    printf("data is uniform random, which is the worst case for quantization recall:\n");
    printf("real embeddings have structure that the quantizers exploit\n\n");

    fprintf(stderr, "building %d x %d table...\n", NVECS, DIM);
    double t0 = now_seconds();
    run_sql(db, "CREATE TABLE t(id INTEGER PRIMARY KEY, v BLOB);");
    run_sql(db, "BEGIN;");
    sqlite3_prepare_v2(db, "INSERT INTO t(id,v) VALUES(?,?);", -1, &stmt, NULL);
    float *row = (float *)malloc(DIM * sizeof(float));
    for (int i = 0; i < NVECS; ++i) {
        for (int j = 0; j < DIM; ++j) row[j] = (float)(rng_unit() * 2.0 - 1.0);
        sqlite3_bind_int(stmt, 1, i + 1);
        sqlite3_bind_blob(stmt, 2, row, (int)(DIM * sizeof(float)), SQLITE_TRANSIENT);
        sqlite3_step(stmt);
        sqlite3_reset(stmt);
        if (((i + 1) % 100000) == 0) fprintf(stderr, "  %d rows\n", i + 1);
    }
    sqlite3_finalize(stmt);
    run_sql(db, "COMMIT;");
    free(row);
    for (int q = 0; q < NQUERIES; ++q) {
        for (int j = 0; j < DIM; ++j) queries[q][j] = (float)(rng_unit() * 2.0 - 1.0);
    }
    fprintf(stderr, "built in %.1fs\n\n", now_seconds() - t0);

    char init_sql[160];
    snprintf(init_sql, sizeof(init_sql), "SELECT vector_init('t','v','type=FLOAT32,dimension=%d,distance=%s');", DIM, DISTANCE);
    run_sql(db, init_sql);

    printf("| mode                     |  MB      |  ms/query |  Mvec/s   | recall |\n");
    printf("|--------------------------|----------|-----------|-----------|--------|\n");

    fprintf(stderr, "exact full scan...\n");
    double exact_time = measure(db, "vector_full_scan", NULL, NULL);
    report("FLOAT32 exact", 0, exact_time, 100.0);

    // max_memory bounds the chunk the scan streams through when the index is not
    // preloaded, and is the whole point of that configuration: the index here is 740 MB,
    // the scan walks it 30 MB at a time
    struct { const char *opts; const char *label; } modes[] = {
        { "qtype=UINT8,max_memory=30MB",         "UINT8" },
        { "qtype=INT8,max_memory=30MB",          "INT8" },
        { "qtype=1BIT,max_memory=30MB",          "1BIT" },
        { "qtype=TURBO,qbits=2,max_memory=30MB", "TURBO2" },
        { "qtype=TURBO,qbits=4,max_memory=30MB", "TURBO4" },
    };

    double int8_stream_ms = 0, int8_pre_ms = 0, int8_recall = 0;
    sqlite3_int64 int8_bytes = 0, int8_stream_mem = 0, int8_pre_mem = 0;

    for (unsigned m = 0; m < sizeof(modes) / sizeof(modes[0]); ++m) {
        char sql[160], label[64];
        fprintf(stderr, "quantizing %s...\n", modes[m].label);
        snprintf(sql, sizeof(sql), "SELECT vector_quantize('t','v','%s');", modes[m].opts);
        run_sql(db, sql);
        sqlite3_int64 bytes = index_bytes(db);

        double recall = 0.0;
        sqlite3_int64 before = sqlite3_memory_used();
        sqlite3_memory_highwater(1);
        double t = measure(db, "vector_quantize_scan", exact, &recall);
        sqlite3_int64 stream_peak = sqlite3_memory_highwater(0) - before;
        snprintf(label, sizeof(label), "%s (30 MB)", modes[m].label);
        report(label, bytes, t, recall);
        if (strcmp(modes[m].label, "INT8") == 0) { int8_stream_ms = t; int8_stream_mem = stream_peak; }

        before = sqlite3_memory_used();
        run_sql(db, "SELECT vector_quantize_preload('t','v');");
        sqlite3_int64 preload_held = sqlite3_memory_used() - before;
        t = measure(db, "vector_quantize_scan", exact, &recall);
        snprintf(label, sizeof(label), "%s preloaded", modes[m].label);
        report(label, bytes, t, recall);

        if (strcmp(modes[m].label, "INT8") == 0) {
            int8_bytes = bytes;
            int8_recall = recall;
            int8_pre_ms = t;
            int8_pre_mem = preload_held;
        }

        run_sql(db, "SELECT vector_quantize_cleanup('t','v');");
    }

    // The README table compares hardware, so it carries the two configurations that
    // differ by deployment rather than by accuracy: the whole index in RAM, and the same
    // index streamed in 30 MB. Everything else about INT8 is a property of the data.
    char nbuf[32];
    with_separators(NVECS, nbuf, sizeof(nbuf));
    printf("\n\nPaste these two rows into the hardware table in README.md:\n\n");
    printf("| %s | %s | `INT8` preloaded | %.0f MB | %.1f | %.1f | %.1f%% |\n",
           HARDWARE, nbuf, (double)int8_pre_mem / (1024.0 * 1024.0),
           int8_pre_ms * 1000.0, NVECS / int8_pre_ms / 1e6, int8_recall);
    printf("| %s | %s | `INT8` streamed | %.0f MB | %.1f | %.1f | %.1f%% |\n",
           HARDWARE, nbuf, (double)int8_stream_mem / (1024.0 * 1024.0),
           int8_stream_ms * 1000.0, NVECS / int8_stream_ms / 1e6, int8_recall);
    printf("\nreference for this machine: FLOAT32 exact %.1f ms/query, index on disk %.0f MB\n",
           exact_time * 1000.0, (double)int8_bytes / (1024.0 * 1024.0));

    sqlite3_close(db);
    return 0;
}

//
//  backend.c
//  sqlitevector
//
//  Reports which distance kernels the extension actually installed, and optionally
//  asserts that it is the expected one. A scan that silently falls back to a lower tier
//  passes every test, so CI needs a way to tell "ran on AVX-512" from "meant to".
//
//    ./backend            print the installed backends
//    ./backend AVX512     print them, and exit non-zero unless the distance backend is AVX512
//

#include <stdio.h>
#include <string.h>

#include "sqlite3.h"

extern int sqlite3_vector_init (sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);

int main (int argc, char **argv) {
    sqlite3 *db = NULL;
    if (sqlite3_open(":memory:", &db) != SQLITE_OK) {
        fprintf(stderr, "unable to open an in-memory database\n");
        return 2;
    }

    int rc = sqlite3_vector_init(db, NULL, NULL);
    if (rc != SQLITE_OK) {
        fprintf(stderr, "sqlite3_vector_init failed (%d)\n", rc);
        sqlite3_close(db);
        return 2;
    }

    sqlite3_stmt *stmt = NULL;
    if (sqlite3_prepare_v2(db, "SELECT vector_backend(), vector_turboquant_backend();", -1, &stmt, NULL) != SQLITE_OK ||
        sqlite3_step(stmt) != SQLITE_ROW) {
        fprintf(stderr, "unable to read the installed backends: %s\n", sqlite3_errmsg(db));
        sqlite3_finalize(stmt);
        sqlite3_close(db);
        return 2;
    }

    const char *distance = (const char *)sqlite3_column_text(stmt, 0);
    const char *turbo = (const char *)sqlite3_column_text(stmt, 1);
    if (!distance) distance = "?";
    if (!turbo) turbo = "?";
    printf("distance backend:   %s\n", distance);
    printf("turboquant backend: %s\n", turbo);

    int result = 0;
    if (argc > 1) {
        result = (strcmp(distance, argv[1]) == 0) ? 0 : 1;
        if (result) fprintf(stderr, "expected the %s backend, but %s was installed\n", argv[1], distance);
    }

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return result;
}

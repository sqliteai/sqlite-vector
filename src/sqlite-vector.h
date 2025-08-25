//
//  sqlite-vector.h
//  sqlitevector
//
//  Created by Marco Bambini on 06/05/25.
//

#ifndef __SQLITE_VECTOR__
#define __SQLITE_VECTOR__

#ifndef SQLITE_CORE
#include "sqlite3ext.h"
#else
#include "sqlite3.h"
#endif

#ifdef _WIN32
  #define SQLITE_VECTOR_API __declspec(dllexport)
#else
  #define SQLITE_VECTOR_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SQLITE_VECTOR_VERSION "0.9.9"

SQLITE_VECTOR_API int sqlite3_vector_init (sqlite3 *db, char **pzErrMsg, const sqlite3_api_routines *pApi);

#ifdef __cplusplus
}
#endif


#endif

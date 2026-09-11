// Copyright (c) 2025 SQLite Cloud, Inc.
// Licensed under the Apache License 2.0 (see LICENSE.md).

import 'dart:ffi';

import 'package:sqlite3/sqlite3.dart';

// @Native resolves from the code asset declared in hook/build.dart.
// The asset ID is 'package:sqlite_vector/src/native/sqlite_vector_extension.dart'.
@Native<Int Function(Pointer<Void>, Pointer<Void>, Pointer<Void>)>(
  assetId: 'package:sqlite_vector/src/native/sqlite_vector_extension.dart',
)
external int sqlite3_vector_init(
  Pointer<Void> db,
  Pointer<Void> pzErrMsg,
  Pointer<Void> pApi,
);

extension SqliteVectorExtension on Sqlite3 {
  /// Loads the sqlite-vector extension.
  ///
  /// Call once at app startup. All subsequently opened databases
  /// will have vector functions available.
  ///
  /// Works with both `sqlite3` package and `drift` ORM.
  void loadSqliteVectorExtension() {
    ensureExtensionLoaded(
      SqliteExtension(
        Native.addressOf<
              NativeFunction<
                Int Function(Pointer<Void>, Pointer<Void>, Pointer<Void>)
              >
            >(sqlite3_vector_init)
            .cast(),
      ),
    );
  }
}

# SQLite Vector Extension – API Reference

This extension enables efficient vector operations directly inside SQLite databases, making it ideal for on-device and edge AI applications. It supports various vector types and SIMD-accelerated distance functions.

### Getting started

* All vectors must have a fixed dimension per column, set during `vector_init`.
* Only tables explicitly initialized using `vector_init` are eligible for vector search.
* You **must run `vector_quantize()`** before using `vector_quantize_scan()`.
* You can preload quantization at database open using `vector_quantize_preload()`.

---

## `vector_version()`

**Returns:** `TEXT`

**Description:**
Returns the current version of the SQLite Vector Extension.

**Example:**

```sql
SELECT vector_version();
-- e.g., '1.0.0'
```

---

## `vector_backend()`

**Returns:** `TEXT`

**Description:**
Returns the active backend used for vector computation. This indicates the SIMD or hardware acceleration available on the current system.

**Possible Values:**

* `CPU` – Generic fallback
* `SSE2` – SIMD on Intel/AMD
* `AVX2` – Advanced SIMD on modern x86 CPUs
* `NEON` – SIMD on ARM (e.g., mobile)

**Example:**

```sql
SELECT vector_backend();
-- e.g., 'AVX2'
```

---

## `vector_init(table, column, options)`

**Returns:** `NULL`

**Description:**
Initializes the vector extension for a given table and column. This is **mandatory** before performing any vector search or quantization.
`vector_init` must be called in every database connection that needs to perform vector operations.

**Parameters:**

* `table` (TEXT): Name of the table containing vector data.
* `column` (TEXT): Name of the column containing the vector embeddings (stored as BLOBs).
* `options` (TEXT): Comma-separated key=value string.

**Options:**

* `dimension` (required): Integer specifying the length of each vector.
* `type`: Vector data type. Options:

  * `FLOAT32` (default)
  * `FLOAT16`
  * `FLOATB16`
  * `INT8`
  * `UINT8`
* `distance`: Distance function to use. Options:

  * `L2` (default)
  * `SQUARED_L2`
  * `COSINE`
  * `DOT`
  * `L1`

**Example:**

```sql
SELECT vector_init('documents', 'embedding', 'dimension=384,type=FLOAT32,distance=cosine');
```

---

## `vector_quantize(table, column, options)`

**Returns:** `INTEGER`

**Description:**
Returns the total number of succesfully quantized rows.

Performs quantization on the specified table and column. This precomputes internal data structures to support fast approximate nearest neighbor (ANN) search.
Read more about quantization [here](https://github.com/sqliteai/sqlite-vector/blob/main/QUANTIZATION.md).

If a quantization already exists for the specified table and column, it is replaced. If it was previously loaded into memory using `vector_quantize_preload`, the data is automatically reloaded. `vector_quantize` should be called once after data insertion. If called multiple times, the previous quantized data is replaced. The resulting quantization is shared across all database connections, so they do not need to call it again.

**Parameters:**

* `table` (TEXT): Name of the table.
* `column` (TEXT): Name of the column containing vector data.
* `options` (TEXT, optional): Comma-separated key=value string.

**Available options:**

* `max_memory`: Max memory to use for quantization (default: 30MB)

**Example:**

```sql
SELECT vector_quantize('documents', 'embedding', 'max_memory=50MB');
```

---

## `vector_quantize_memory(table, column)`

**Returns:** `INTEGER`

**Description:**
Returns the amount of memory (in bytes) required to preload quantized data for the specified table and column.

**Example:**

```sql
SELECT vector_quantize_memory('documents', 'embedding');
-- e.g., 28490112
```

---

## `vector_quantize_preload(table, column)`

**Returns:** `NULL`

**Description:**
Loads the quantized representation for the specified table and column into memory. Should be used at startup to ensure optimal query performance.
`vector_quantize_preload` should be called once after `vector_quantize`. The preloaded data is also shared across all database connections, so they do not need to call it again.

**Example:**

```sql
SELECT vector_quantize_preload('documents', 'embedding');
```

---

## `vector_quantize_cleanup(table, column)`

**Returns:** `NULL`

**Description:**
Releases memory previously allocated by a `vector_quantize_preload` call and removes all quantization entries associated with the specified table and column.
Use this function when quantization is no longer required. In some cases, running VACUUM may be necessary to reclaim the freed space from the database.

If the data changes and you invoke `vector_quantize`, the existing quantization data is automatically replaced. In that case, calling this function is unnecessary.

**Example:**

```sql
SELECT vector_quantize_cleanup('documents', 'embedding');
```

---

## `vector_as_f32(value)`

## `vector_as_f16(value)`

## `vector_as_bf16(value)`

## `vector_as_i8(value)`

## `vector_as_u8(value)`

**Returns:** `BLOB`

**Description:**
Encodes a vector into the required internal BLOB format to ensure correct storage and compatibility with the system’s vector representation.
A real conversion is performed ONLY in case of JSON input. When input is a BLOB, it is assumed to be already properly formatted.

Functions in the `vector_as_` family should be used in all `INSERT`, `UPDATE`, and `DELETE` statements to properly format vector values. However, they are *not* required when specifying input vectors for the `vector_full_scan` or `vector_quantize_scan` virtual tables.

**Parameters:**

* `value` (TEXT or BLOB):

  * If `TEXT`, it must be a JSON array (e.g., `"[0.1, 0.2, 0.3]"`).
  * If `BLOB`, no check is performed; the user must ensure the format matches the specified type and dimension.

* `dimension` (INT, optional): Enforce a stricter sanity check, ensuring the input vector has the expected dimensionality.

**Usage by format:**

```sql
-- Insert a Float32 vector using JSON
INSERT INTO documents(embedding) VALUES(vector_as_f32('[0.1, 0.2, 0.3]'));

-- Insert a UInt8 vector using raw BLOB (ensure correct formatting!)
INSERT INTO compressed_vectors(embedding) VALUES(vector_as_u8(X'010203'));
```

---

## 🔍 `vector_full_scan(table, column, vector, k)`

**Returns:** `Virtual Table (rowid, distance)`

**Description:**
Performs a brute-force nearest neighbor search using the given vector. Despite its brute-force nature, this function is highly optimized and useful for small datasets or validation.

**Parameters:**

* `table` (TEXT): Name of the target table.
* `column` (TEXT): Column containing vectors.
* `vector` (BLOB or JSON): The query vector.
* `k` (INTEGER): Number of nearest neighbors to return.

**Example:**

```sql
SELECT rowid, distance
FROM vector_full_scan('documents', 'embedding', vector_as_f32('[0.1, 0.2, 0.3]'), 5);
```

---

## ⚡ `vector_quantize_scan(table, column, vector, k)`

**Returns:** `Virtual Table (rowid, distance)`

**Description:**
Performs a fast approximate nearest neighbor search using the pre-quantized data. This is the **recommended query method** for large datasets due to its excellent speed/recall/memory trade-off.

You **must run `vector_quantize()`** before using `vector_quantize_scan()` and when data initialized for vectors changes.

**Parameters:**

* `table` (TEXT): Name of the target table.
* `column` (TEXT): Column containing vectors.
* `vector` (BLOB or JSON): The query vector.
* `k` (INTEGER): Number of nearest neighbors to return.

**Performance Highlights:**

* Handles **1M vectors** of dimension 768 in a few milliseconds.
* Uses **<50MB** of RAM.
* Achieves **>0.95 recall**.

**Example:**

```sql
SELECT rowid, distance
FROM vector_quantize_scan('documents', 'embedding', vector_as_f32('[0.1, 0.2, 0.3]'), 10);
```

---

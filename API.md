# SQLite Vector Extension – API Reference

This extension enables efficient vector operations directly inside SQLite databases, making it ideal for on-device and edge AI applications. It supports various vector types and SIMD-accelerated distance functions.

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

**Returns:** `NULL`

**Description:**
Performs quantization on the specified table and column. This precomputes internal data structures to support fast approximate nearest neighbor (ANN) search.

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

**Example:**

```sql
SELECT vector_quantize_preload('documents', 'embedding');
```

---

## `vector_cleanup(table, column)`

**Returns:** `NULL`

**Description:**
Cleans up internal structures related to a previously quantized table/column. Use this if data has changed or quantization is no longer needed.

**Example:**

```sql
SELECT vector_cleanup('documents', 'embedding');
```

---

## `vector_convert_f32(value)`

## `vector_convert_f16(value)`

## `vector_convert_bf16(value)`

## `vector_convert_i8(value)`

## `vector_convert_u8(value)`

**Returns:** `BLOB`

**Description:**
Encodes a vector into the required internal BLOB format to ensure correct storage and compatibility with the system’s vector representation.

Functions in the `vector_convert_` family should be used in all `INSERT`, `UPDATE`, and `DELETE` statements to properly format vector values. However, they are *not* required when specifying input vectors for the `vector_full_scan` or `vector_quantize_scan` virtual tables.

Optionally, these functions accept a `dimension INT` argument (placed before the vector value) to enforce a stricter sanity check, ensuring the input vector has the expected dimensionality.

**Parameters:**

* `value` (TEXT or BLOB):

  * If `TEXT`, it must be a JSON array (e.g., `"[0.1, 0.2, 0.3]"`).
  * If `BLOB`, no check is performed; the user must ensure the format matches the specified type and dimension.

**Usage by format:**

```sql
-- Insert a Float32 vector using JSON
INSERT INTO documents(embedding) VALUES(vector_convert_f32('[0.1, 0.2, 0.3]'));

-- Insert a UInt8 vector using raw BLOB (ensure correct formatting!)
INSERT INTO compressed_vectors(embedding) VALUES(vector_convert_u8(X'010203'));
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
FROM vector_full_scan('documents', 'embedding', vector_convert_f32('[0.1, 0.2, 0.3]'), 5);
```

---

## ⚡ `vector_quantize_scan(table, column, vector, k)`

**Returns:** `Virtual Table (rowid, distance)`

**Description:**
Performs a fast approximate nearest neighbor search using the pre-quantized data. This is the **recommended query method** for large datasets due to its excellent speed/recall/memory trade-off.

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
FROM vector_quantize_scan('documents', 'embedding', vector_convert_f32('[0.1, 0.2, 0.3]'), 10);
```

---

## 📌 Notes

* All vectors must have a fixed dimension per column, set during `vector_init`.
* Only tables explicitly initialized using `vector_init` are eligible for vector search.
* You **must run `vector_quantize()`** before using `vector_quantize_scan()`.
* You can preload quantization at database open using `vector_quantize_preload()`.

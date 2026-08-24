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
* `AVX512` – Wide SIMD on supported x86 CPUs
* `NEON` – SIMD on ARM (e.g., mobile)
* `RVV` – SIMD on supported RISC-V CPUs

**Example:**

```sql
SELECT vector_backend();
-- e.g., 'AVX2'
```

---

## `vector_turboquant_backend()`

**Returns:** `TEXT`

**Description:**
Returns the SIMD tier selected at load time, the same one `vector_backend()` reports.

TurboQuant lookup-table scans no longer vary by backend: the scan is one table lookup
per row, which is already about one load per cycle on any machine, and NEON has no
gather instruction at all. A single implementation is used everywhere, so the same query
returns the same distance whatever the CPU — the per-backend versions this replaced
differed by up to 1.5e-4 relative because they accumulated in `float` while the scalar
one accumulated in `double`.

**Example:**

```sql
SELECT vector_turboquant_backend();
-- e.g., 'NEON'
```

---

## `vector_init(table, column, options)`

**Returns:** `NULL`

**Description:**
Initializes the vector extension for a given table and column. This is **mandatory** before performing any vector search or quantization.
`vector_init` must be called in every database connection that needs to perform vector operations.

The target table must have a **`rowid`** (an integer primary key, either explicit or implicit).
If the table was created using `WITHOUT ROWID`, it must have **exactly one primary key column of type `INTEGER`**.
This ensures that each vector can be uniquely identified and efficiently referenced during search and quantization.

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
  * `1BIT`
* `distance`: Distance function to use. Options:

  * `L2` (default)
  * `SQUARED_L2`
  * `COSINE`
  * `DOT`
  * `L1`
  * `HAMMING` (only valid with `type=1BIT`)
* `normalized`: Set to `1` to declare that every stored vector is unit length. With
  `type=FLOAT32` and `distance=COSINE` this lets a full-precision scan compute
  `1 - dot` instead of the full cosine, dropping two thirds of the arithmetic from the
  inner loop; the query vector is normalized once per scan, so the reported distances are
  unchanged. It is an assertion, not a request: if the stored vectors are *not* unit
  length the distances will be wrong. Quantized scans ignore it, because the quantized
  index holds scaled integers whose norm is not 1. Default `0`.

**Example:**

```sql
SELECT vector_init('documents', 'embedding', 'dimension=384,type=FLOAT32,distance=cosine');

-- embeddings already normalized by the model: faster cosine, same results
SELECT vector_init('documents', 'embedding', 'dimension=384,type=FLOAT32,distance=cosine,normalized=1');
```

---

## `vector_quantize(table, column, options)`

**Returns:** `INTEGER`

**Description:**
Returns the total number of successfully quantized rows.

Performs quantization on the specified table and column. This precomputes internal data structures to support fast approximate nearest neighbor (ANN) search.
Read more about quantization [here](https://github.com/sqliteai/sqlite-vector/blob/main/QUANTIZATION.md).

If a quantization already exists for the specified table and column, it is replaced. If it was previously loaded into memory using `vector_quantize_preload`, the data is automatically reloaded. `vector_quantize` should be called once after data insertion. If called multiple times, the previous quantized data is replaced. The resulting quantization is shared across all database connections, so they do not need to call it again.

**Parameters:**

* `table` (TEXT): Name of the table.
* `column` (TEXT): Name of the column containing vector data.
* `options` (TEXT, optional): Comma-separated key=value string.

**Available options:**

* `max_memory`: Max memory to use for quantization (default: 30MB)
* `qtype`: Quantization type: `UINT8`, `INT8`, `1BIT`, `TURBO`, `TURBO2`, `TURBO3`, or `TURBO4`
* `qbits`: TurboQuant bit width (`2`, `3`, or `4`). Defaults to `4` when `qtype=TURBO`.

**Example:**

```sql
SELECT vector_quantize('documents', 'embedding', 'max_memory=50MB');
SELECT vector_quantize('documents', 'embedding', 'qtype=BIT');
SELECT vector_quantize('documents', 'embedding', 'qtype=TURBO,qbits=4');
SELECT vector_quantize('documents', 'embedding', 'qtype=TURBO2');
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

## `vector_as_bit(value)`

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

## 🔍 `vector_full_scan(table, column, vector [, k])`

**Returns:** `Virtual Table (rowid, distance)`

**Description:**
Performs a brute-force nearest neighbor search using the given vector. Despite its brute-force nature, this function is highly optimized and useful for small datasets (rows < 1000000) or validation.
Since this interface only returns rowid and distance, if you need to access additional columns from the original table, you must use a SELF JOIN.

**Parameters:**

* `table` (TEXT): Name of the target table.
* `column` (TEXT): Column containing vectors.
* `vector` (BLOB or JSON): The query vector.
* `k` (INTEGER, optional): Number of nearest neighbors to return. When provided, the module collects the top-k results sorted by distance. When omitted, the module operates in **streaming mode** — rows are returned progressively as they are scanned, enabling standard SQL clauses such as `WHERE` and `LIMIT` to control filtering and result count.

**Examples:**

```sql
-- Top-k mode: return the 5 nearest neighbors, sorted by distance
SELECT rowid, distance
FROM vector_full_scan('documents', 'embedding', vector_as_f32('[0.1, 0.2, 0.3]'), 5);
```

```sql
-- Streaming mode: progressively scan all rows, apply SQL filters
SELECT rowid, distance
FROM vector_full_scan('documents', 'embedding', vector_as_f32('[0.1, 0.2, 0.3]'))
LIMIT 5;
```

```sql
-- Streaming mode with JOIN and filtering
SELECT
    v.rowid,
    row_number() OVER (ORDER BY v.distance) AS rank_number,
    v.distance
FROM vector_full_scan('documents', 'embedding', vector_as_f32('[0.1, 0.2, 0.3]')) AS v
    JOIN documents ON documents.rowid = v.rowid
WHERE documents.category = 'science'
LIMIT 10;
```

---

## ⚡ `vector_quantize_scan(table, column, vector [, k])`

**Returns:** `Virtual Table (rowid, distance)`

**Description:**
Performs a fast approximate nearest neighbor search using the pre-quantized data. This is the **recommended query method** for large datasets due to its excellent speed/recall/memory trade-off. Since this interface only returns rowid and distance, if you need to access additional columns from the original table, you must use a SELF JOIN.

You **must run `vector_quantize()`** before using `vector_quantize_scan()` and when data initialized for vectors changes.

**Parameters:**

* `table` (TEXT): Name of the target table.
* `column` (TEXT): Column containing vectors.
* `vector` (BLOB or JSON): The query vector.
* `k` (INTEGER, optional): Number of nearest neighbors to return. When provided, the module collects the top-k results sorted by distance. When omitted, the module operates in **streaming mode** — rows are returned progressively, enabling standard SQL clauses such as `WHERE` and `LIMIT`.

**Performance Highlights:**

* Supports compact SIMD 2-, 3-, and 4-bit TurboQuant scans for high-dimensional vectors.
* `qbits=2` minimizes memory; `qbits=4` usually gives the better recall/speed balance.
* Recall depends on the dataset, distance function, bit width, and `k`; validate it against `vector_full_scan()` for your workload.

**Examples:**

```sql
-- Top-k mode: return the 10 nearest neighbors, sorted by distance
SELECT rowid, distance
FROM vector_quantize_scan('documents', 'embedding', vector_as_f32('[0.1, 0.2, 0.3]'), 10);
```

```sql
-- Streaming mode: progressively scan using quantized data
SELECT rowid, distance
FROM vector_quantize_scan('documents', 'embedding', vector_as_f32('[0.1, 0.2, 0.3]'))
LIMIT 10;
```

**Usage Notes:**

* In **top-k mode** (with `k`), results are sorted by distance. The query planner knows the output is pre-sorted, so no additional `ORDER BY` is needed.
* In **streaming mode** (without `k`), rows are returned in scan order. Use `ORDER BY distance` and `LIMIT` as needed.
* Streaming mode is ideal for combining vector similarity with additional SQL-level filters or progressive result consumption.

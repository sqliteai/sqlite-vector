### Vector Quantization for High Performance

`sqlite-vector` supports **vector quantization**, a powerful technique to significantly accelerate vector search while reducing memory usage. You can quantize your vectors with:

```sql
SELECT vector_quantize('my_table', 'my_column');
```

To further boost performance, quantized vectors can be **preloaded in memory** using:

```sql
SELECT vector_quantize_preload('my_table', 'my_column');
```

This can result in a **4×–5× speedup** on nearest neighbor queries while keeping memory usage low.

#### What is Quantization?

Quantization compresses high-dimensional float vectors (e.g., `FLOAT32`) into compact representations using lower-precision formats (e.g., `UINT8`). This drastically reduces the size of the data—often by a factor of 4 to 8—making it practical to load large datasets entirely in memory, even on edge devices.

#### Why is it Important?

* **Faster Searches**: With preloaded quantized vectors, distance computations are up to 5× faster.
* **Lower Memory Footprint**: Quantized vectors use significantly less RAM, allowing millions of vectors to fit in memory.
* **Edge-ready**: The reduced size and in-memory access make this ideal for mobile, embedded, and on-device AI applications.

#### Estimate Memory Usage

Before preloading quantized vectors, you can **estimate the memory required** using:

```sql
SELECT vector_quantize_memory('my_table', 'my_column');
```

This gives you an approximate number of bytes needed to load the quantized vectors into memory.

#### Accuracy You Can Trust

Despite the compression, our quantization algorithms are finely tuned to maintain high accuracy. You can expect **recall rates greater than 0.95**, ensuring that approximate searches closely match exact results in quality.

#### Measuring Recall in SQLite-Vector

You can evaluate the recall of quantized search compared to exact search using a single SQL query. For example, assuming a table `vec_examples` with an `embedding` column, use:

```sql
WITH
exact_knn AS (
    SELECT e.rowid
    FROM vec_examples AS e
    JOIN vector_full_scan('vec_examples', 'embedding', ?1, ?2) AS v
    ON e.rowid = v.rowid
),
approx_knn AS (
    SELECT e.rowid
    FROM vec_examples AS e
    JOIN vector_quantize_scan('vec_examples', 'embedding', ?1, ?2) AS v
    ON e.rowid = v.rowid
),
matches AS (
    SELECT COUNT(*) AS match_count
    FROM exact_knn
    WHERE rowid IN (SELECT rowid FROM approx_knn)
),
total AS (
    SELECT COUNT(*) AS total_count
    FROM exact_knn
)
SELECT
    (SELECT match_count FROM matches) AS match_count,
    (SELECT total_count FROM total) AS total_count,
    CAST((SELECT match_count FROM matches) AS FLOAT) /
    CAST((SELECT total_count FROM total) AS FLOAT) AS recall;
```

Where `?1` is the input vector (as a BLOB) and `?2` is the number of nearest neighbors `k`.
This query compares exact and quantized results and computes the recall ratio, helping you validate the quality of quantized search.

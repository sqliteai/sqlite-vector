# SQLite Vector

**SQLite Vector** is a cross-platform, ultra-efficient SQLite extension that brings vector search capabilities to your embedded database. It works seamlessly on **iOS, Android, Windows, Linux, and macOS**, using just **30MB of memory** by default. With support for **Float32, Float16, BFloat16, Int8, and UInt8**, and **highly optimized distance functions**, it's the ideal solution for **Edge AI** applications.

## 🚀 Highlights

* ✅ **No virtual tables required** – store vectors directly as `BLOB`s in ordinary tables
* ✅ **Blazing fast** – optimized C implementation with SIMD acceleration
* ✅ **Low memory footprint** – defaults to just 30MB of RAM usage
* ✅ **Zero preindexing needed** – no long preprocessing or index-building phases
* ✅ **Works offline** – perfect for on-device, privacy-preserving AI workloads
* ✅ **Plug-and-play** – drop into existing SQLite workflows with minimal effort
* ✅ **Cross-platform** – works out of the box on all major OSes


## Why Use SQLite-Vector?

| Feature                    | SQLite-Vector | Traditional Solutions                      |
| -------------------------- | ------------- | ------------------------------------------ |
| Works with ordinary tables | ✅             | ❌ (usually require special virtual tables) |
| Requires preindexing       | ❌             | ✅ (can take hours for large datasets)      |
| Requires external server   | ❌             | ✅ (often needs Redis/FAISS/Weaviate/etc.)  |
| Memory-efficient           | ✅             | ❌                                          |
| Easy to use SQL            | ✅             | ❌ (often complex JOINs, subqueries)        |
| Offline/Edge ready         | ✅             | ❌                                          |
| Cross-platform             | ✅             | ❌                                          |

Unlike other vector databases or extensions that require complex setup, SQLite-Vector **just works** with your existing database schema and tools.


## 📦 Installation

### Pre-built Binaries

Download the appropriate pre-built binary for your platform from the official [Releases](https://github.com/sqliteai/sqlite-vector/releases) page:

- Linux: x86 and ARM
- macOS: x86 and ARM
- Windows: x86
- Android
- iOS

### Loading the Extension

```sql
-- In SQLite CLI
.load ./vector

-- In SQL
SELECT load_extension('./vector');
```

Or embed it directly into your application.

## Example Usage

```sql
-- Create a regular SQLite table
CREATE TABLE images (
  id INTEGER PRIMARY KEY,
  embedding BLOB, -- store Float32/UInt8/etc.
  label TEXT
);

-- Insert a BLOB vector (Float32, 384 dimensions) using bindings
INSERT INTO images (embedding, label) VALUES (?, 'cat');

-- Insert a JSON vector (Float32, 384 dimensions)
INSERT INTO images (embedding, label) VALUES (vector_convert_f32('[0.3, 1.0, 0.9, 3.2, 1.4,...]'), 'dog');

-- Initialize the vector. By default, the distance function is L2.
-- To use a different metric, specify one of the following options:
-- distance=L1, distance=COSINE, distance=DOT, or distance=SQUARED_L2.
SELECT vector_init('images', 'embedding', 'type=FLOAT32,dimension=384');

-- Quantize vector
SELECT vector_quantize('images', 'embedding');

-- Optional preload quantized version in memory (for a 4x/5x speedup) 
SELECT vector_quantize_preload('images', 'embedding');

-- Run a nearest neighbor query on the quantized version (returns top 20 closest vectors)
SELECT e.id, v.distance FROM images AS e
   JOIN vector_quantize_scan('images', 'embedding', ?, 20) AS v
   ON e.id = v.rowid;
```

## 📋 Documentation

Extensive API documentation can be found in the [API page](https://github.com/sqliteai/sqlite-vector/blob/main/API.md).

More information about the quantization process can be found in the [QUANTIZATION document](https://github.com/sqliteai/sqlite-vector/blob/main/QUANTIZATION.md).

## 🧩 Features

### Instant Vector Search – No Preindexing Required

Unlike other SQLite vector extensions that rely on complex indexing algorithms such as DiskANN, HNSW, or IVF, which often require **preprocessing steps that can take hours or even days**, `sqlite-vector` works out of the box with your existing data. There’s **no need to preindex your vectors**—you can start performing fast, approximate or exact vector searches **immediately**.

This means:

* 🕒 **No waiting time** before your app or service is usable
* 🔄 **Zero-cost updates** – you can add, remove, or modify vectors on the fly without rebuilding any index
* ⚡ **Works directly with BLOB columns** in ordinary SQLite tables – no special schema or virtual table required
* 🌍 **Ideal for edge and mobile use cases**, where preprocessing large datasets is not practical or possible

By eliminating the need for heavyweight indexing, `sqlite-vector` offers a **simpler, faster, and more developer-friendly** approach to embedding vector search in your applications.

### Supported Vector Types

You can store your vectors as `BLOB` columns in ordinary tables. Supported formats include:

* `float32` (4 bytes per element)
* `float16` (2 bytes per element)
* `bfloat16` (2 bytes per element)
* `int8` (1 byte per element)
* `uint8` (1 byte per element)

Simply insert a vector as a binary blob into your table. No special table types or schemas are required.


### Supported Distance Metrics

Optimized implementations available:

* **L2 Distance (Euclidean)**
* **Squared L2**
* **L1 Distance (Manhattan)**
* **Cosine Distance**
* **Dot Product**

These are implemented in pure C and optimized for SIMD when available, ensuring maximum performance on modern CPUs and mobile devices.

---

# 🧠 What Is Vector Search?

Vector search is the process of finding the closest match(es) to a given vector (a point in high-dimensional space) based on a similarity or distance metric. It is essential for AI and machine learning applications where data is often encoded into vector embeddings.

### Common Use Cases

* **Semantic Search**: find documents, emails, or messages similar to a query
* **Image Retrieval**: search for visually similar images
* **Recommendation Systems**: match users with products, videos, or music
* **Voice and Audio Search**: match voice queries or environmental sounds
* **Anomaly Detection**: find outliers in real-time sensor data
* **Robotics**: localize spatial features or behaviors using embedded observations

In the AI era, embeddings are everywhere – from language models like GPT to vision transformers. Storing and searching them efficiently is the foundation of intelligent applications.

## 🌍 Perfect for Edge AI

SQLite-Vector is designed with the **Edge AI** use case in mind:

* 📴 Runs offline – no internet required
* 📱 Works on mobile devices – iOS/Android friendly
* 🔒 Keeps data local – ideal for privacy-focused apps
* ⚡ Extremely fast – real-time performance on device

You can deploy powerful similarity search capabilities right inside your app or embedded system – **no cloud needed**.

## License

This project is licensed under the [Elastic License 2.0](./LICENSE.md). You can use, copy, modify, and distribute it under the terms of the license for non-production use. For production or managed service use, please [contact SQLite Cloud, Inc](mailto:info@sqlitecloud.io) for a commercial license.

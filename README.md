<div align="center">
  <a href="https://sqlite.ai">
    <img src="https://www.sqlite.ai/social/logo-ai.png" alt="SQLite AI" height="56">
  </a>

  <h1>SQLite-Vector</h1>
  <p><strong>Production-grade vector search inside SQLite.</strong><br>
  Exact search, SIMD distance kernels, and SIMD 2/3/4-bit TurboQuant scans — runs anywhere SQLite runs: mobile, browser, edge, server.</p>

  <p>
    <a href="https://dashboard.sqlitecloud.io/auth/sign-in"><strong>Free managed instance →</strong></a> ·
    <a href="https://docs.sqlitecloud.io/docs/ai-overview">Docs</a> ·
    <a href="https://sqlite.ai">Website</a> ·
    <a href="https://blog.sqlite.ai">Blog</a>
  </p>

  <p>
    <sub><strong>Data:</strong>
    <a href="https://github.com/sqliteai/sqlite-vector">Vector</a> ·
    <a href="https://github.com/sqliteai/sqlite-sync">Sync</a> ·
    <a href="https://github.com/sqliteai/sqlite-columnar">Columnar</a> ·
    <a href="https://github.com/sqliteai/sqlite-js">JS</a>
    <br>
    <strong>AI:</strong>
    <a href="https://github.com/sqliteai/sqlite-ai">AI</a> ·
    <a href="https://github.com/sqliteai/sqlite-agent">Agent</a> ·
    <a href="https://github.com/sqliteai/sqlite-memory">Memory</a> ·
    <a href="https://github.com/sqliteai/sqlite-mcp">MCP</a>
    </sub>
  </p>
</div>

<br>

> **Building RAG or semantic search?** SQLite-Vector ships as an extension you can drop into any SQLite app. Need it managed with sync and auth? **[SQLite Cloud free tier](https://dashboard.sqlitecloud.io/auth/sign-in)** gives you 512 MB and 20 connections, no credit card.

---

# SQLite Vector

**SQLite Vector** is a cross-platform, ultra-efficient SQLite extension that brings vector search capabilities to your embedded database. It works seamlessly on **iOS, Android, Windows, Linux, and macOS**, using just **30MB of memory** by default. With support for **Float32, Float16, BFloat16, Int8, UInt8, 1Bit, and TurboQuant 2/3/4-bit quantization**, plus **highly optimized distance functions**, it's the ideal solution for **Edge AI** applications.

SQLite-Vector includes **TurboQuant**, a compact data-oblivious vector quantizer inspired by the Google Research paper [TurboQuant: Online Vector Quantization with Near-Optimal Distortion Rate](https://arxiv.org/abs/2504.19874). It stores each vector as low-bit scalar codes plus one scale value, then scores directly from SIMD lookup-table kernels without reconstructing full vectors.

## Highlights

* **No virtual tables required** – store vectors directly as `BLOB`s in ordinary tables
* **Blazing fast** – optimized C implementation with SIMD acceleration
* **TurboQuant support** – SIMD 2-, 3-, and 4-bit quantization scans with `qtype=TURBO`
* **Low memory footprint** – defaults to just 30MB of RAM usage
* **Zero preindexing needed** – no long preprocessing or index-building phases
* **Works offline** – perfect for on-device, privacy-preserving AI workloads
* **Plug-and-play** – drop into existing SQLite workflows with minimal effort
* **Cross-platform** – works out of the box on all major OSes


## Why Use SQLite-Vector?

| Feature                      | SQLite-Vector | Traditional Solutions                      |
| ---------------------------- | ------------- | ------------------------------------------ |
| Works with ordinary tables   | ✅             | ❌ (usually require special virtual tables) |
| Doesn't need preindexing     | ✅             | ❌ (can take hours for large datasets)      |
| Doesn't need external server | ✅             | ❌ (often needs Redis/FAISS/Weaviate/etc.)  |
| Memory-efficient             | ✅             | ❌                                          |
| TurboQuant low-bit scanning  | ✅             | ❌                                          |
| Easy to use SQL              | ✅             | ❌ (often complex JOINs, subqueries)        |
| Offline/Edge ready           | ✅             | ❌                                          |
| Cross-platform               | ✅             | ❌                                          |

Unlike other vector databases or extensions that require complex setup, SQLite-Vector **just works** with your existing database schema and tools.


## Installation

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

### WASM Version

You can download the WebAssembly (WASM) version of SQLite with the SQLite Vector extension enabled from: https://www.npmjs.com/package/@sqliteai/sqlite-wasm

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
INSERT INTO images (embedding, label) VALUES (vector_as_f32('[0.3, 1.0, 0.9, 3.2, 1.4,...]'), 'dog');

-- Initialize the vector. By default, the distance function is L2.
-- To use a different metric, specify one of the following options:
-- distance=L1, distance=COSINE, distance=DOT, distance=SQUARED_L2, or distance=HAMMING.
SELECT vector_init('images', 'embedding', 'type=FLOAT32,dimension=384');

-- If your embeddings are already unit length, say so: FLOAT32 cosine scans then compute
-- 1 - dot instead of the full cosine, with the same results.
-- SELECT vector_init('images', 'embedding', 'type=FLOAT32,dimension=384,distance=COSINE,normalized=1');

-- Quantize vector
SELECT vector_quantize('images', 'embedding');

-- Or use TurboQuant for compact 2/3/4-bit quantization
SELECT vector_quantize('images', 'embedding', 'qtype=TURBO,qbits=4');

-- Optional preload quantized version in memory (for a 4x/5x speedup) 
SELECT vector_quantize_preload('images', 'embedding');

-- Run a nearest neighbor query on the quantized version (returns top 20 closest vectors)
SELECT e.id, v.distance FROM images AS e
   JOIN vector_quantize_scan('images', 'embedding', ?, 20) AS v
   ON e.id = v.rowid;

-- Streaming mode: omit k to get rows progressively, use SQL to filter and limit
SELECT e.id, v.distance FROM images AS e
   JOIN vector_quantize_scan('images', 'embedding', ?) AS v
   ON e.id = v.rowid
   WHERE e.label = 'cat'
   LIMIT 10;
```

## Benchmark

One command, so results from different machines are comparable:

```bash
make benchmark HARDWARE="Apple M5 Pro - NEON"
```

It builds `test/benchmark.c` at `-O3` with the same per-translation-unit SIMD flags the
shipped extension uses, then runs **k=20 over 1,000,000 vectors of dimension 768** with
cosine distance, 20 queries, reporting the best. It prints the two rows below ready to
paste. Override anything:

```bash
make benchmark NVECS=100000 DIM=384 K=10 DISTANCE=l2
```

Common to every row: vectors are **uniform random** with a fixed seed, so two machines
measure the same data; the `INT8` index is **740 MB** on disk against 2930 MB of raw
`FLOAT32`; recall is the overlap with the exact `FLOAT32` scan, which is the baseline
everything is compared against and is 100% by definition. On the reference machine that
exact scan takes **148.3 ms/query**.

The two rows per machine are the two ways the same index gets deployed. *Preloaded* holds
it entirely in RAM after `vector_quantize_preload()`. *Streamed* walks it through a bounded
buffer set by `max_memory=30MB`, which is the default and what a device with 740 MB of
index and less RAM than that actually does. The **Max memory** column is measured, not the
parameter echoed back: it is what the extension had allocated at the peak of the scan.

### Hardware

| Hardware | Vectors | Index | Max memory | ms/query | Mvec/s | Recall@20 |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Apple M5 Pro - NEON | 1,000,000 | `INT8` preloaded | 740 MB | 37.3 | 26.8 | 99.5% |
| Apple M5 Pro - NEON | 1,000,000 | `INT8` streamed | 30 MB | 56.2 | 17.8 | 99.5% |

*Results from other CPUs welcome — run the command above and open a PR adding your two
rows.*

The trade is 25x less memory for 1.5x the latency, and recall is untouched because both
rows read the same index — only how much of it is resident differs.

Recall is repeated per row on purpose: it depends on the data, not the hardware, so a row
that disagrees with the others is a sign that machine selected a different SIMD backend
than it should have.

### Choosing a mode

`INT8` is in the table because it is the mode to reach for first. The others, measured on
the same machine and data:

| Mode | Index | ms/query | Recall@20 |
| --- | ---: | ---: | ---: |
| `FLOAT32` exact | 2930 MB | 148.3 | 100.0% |
| `UINT8` preloaded | 740 MB | 37.2 | 33.8% |
| `INT8` preloaded | 740 MB | 37.3 | 99.5% |
| `1BIT` preloaded | 99 MB | 2.4 | 10.0% |
| `TURBO2` preloaded | 195 MB | 47.1 | 45.2% |
| `TURBO4` preloaded | 378 MB | 151.3 | 81.8% |

**The data is uniform random**, the worst case for every quantizer: real embeddings have
structure quantization exploits, so recall on your own vectors will be higher, often much
higher. Read that column as a floor and a way to rank the modes, not as a prediction.

Three things are worth knowing before choosing.

**For cosine, use `INT8`, not `UINT8`.** Same size, same speed, 33.8% recall against 99.5%.
Unsigned quantization subtracts the dataset minimum before scaling, and cosine measures
angle, which that shift destroys. `UINT8` is right for L2, where a common translation
cancels. If you omit `qtype` the extension picks `UINT8` for non-negative data — correct
for L2, wrong for cosine — so set it explicitly when you use cosine.

**`1BIT` is a filter, not an answer.** 409 Mvec/s and 30x less memory, at 10% recall here.
It earns its place as a first pass whose survivors you re-rank at full precision.

**TurboQuant trades size, not speed.** `TURBO4` is *slower* than the exact scan while using
8x less memory. Its lookup scan is one table gather per row — at dimension 768 that is 384
gathers into a 384 KB table per vector, already about one lookup per cycle, so the current
storage layout has no headroom left ([#57](https://github.com/sqliteai/sqlite-vector/issues/57)).
Choose TurboQuant when memory is what binds; choose `INT8` when throughput is.

## TurboQuant Benchmark and Recall

TurboQuant can be selected with `qtype=TURBO,qbits=N`, where `N` is `2`, `3`, or `4`. Shorthand aliases are also available: `TURBO2`, `TURBO3`, and `TURBO4`.

```sql
-- Highest recall TurboQuant mode currently recommended as the default
SELECT vector_quantize('images', 'embedding', 'qtype=TURBO,qbits=4');

-- Smaller edge-oriented representation
SELECT vector_quantize('images', 'embedding', 'qtype=TURBO2');
```

An earlier synthetic benchmark reported speedups of 15x for 4-bit and 38x for 2-bit
against `vector_full_scan()`. Those numbers were measured with a **file-backed** database,
where the full scan reads 3 GB of raw vectors off disk and the comparison is dominated by
I/O rather than by arithmetic — and before the distance kernels were rewritten, which made
the full-precision scan itself substantially faster. Against an in-memory baseline on
current code the picture is different: see [Benchmark](#benchmark) below, where `TURBO4`
is marginally slower than the exact scan and its argument is memory, not speed. Both
measurements are real; they answer different questions. If your working set does not fit
in RAM, the file-backed comparison is the one that describes your deployment.

For comparison, the raw `FLOAT32` vectors alone are about **3.07 GB** for 1M x 768 before SQLite row/page overhead. TurboQuant 4-bit reduces the scan representation to about **13%** of that raw vector payload, TurboQuant 3-bit to about **10%**, and TurboQuant 2-bit to about **7%**. Actual resident memory depends on whether the database is in-memory or file-backed, SQLite cache settings, preloading, page cache behavior, and the host allocator.

The TurboQuant scan backend can be checked separately from the regular distance backend:

```sql
SELECT vector_backend(), vector_turboquant_backend();
```

For edge deployments, `vector_quantize_memory(table, column)` estimates the quantized scan representation. TurboQuant stores each row as `rowid + scale + packed_codes`, roughly `rows * (8 + 4 + ceil(dim * qbits / 8))` bytes before allocator and SQLite cache overhead. The synthetic benchmark in `test/benchmark_turboquant.c` also supports `PRELOAD=0` to compare the lower-RAM, non-preloaded path.

Real-dataset recall can be reproduced with `test/recall_turboquant_real.py`, which downloads Fashion-MNIST in the ANN-Benchmarks HDF5 format and compares TurboQuant against `vector_full_scan()` using L2 distance. Example run on macOS ARM64/NEON with 10,000 base vectors, 50 queries, and k=10:

| Mode | Quantized storage | Full scan / query | TurboQuant / query | Speedup | Recall@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| TurboQuant 4-bit | 4.04 MB | 16.32 ms | 4.80 ms | 3.40x | 0.948 |
| TurboQuant 3-bit | 3.06 MB | 16.32 ms | 8.28 ms | 1.97x | 0.868 |
| TurboQuant 2-bit | 2.08 MB | 16.32 ms | 1.86 ms | 8.78x | 0.596 |

`qbits=4` is the recommended starting point when recall matters. `qbits=2` is useful for tighter edge memory budgets, but should be validated on the target embeddings because recall can drop significantly depending on the dataset.

### Swift Package

You can [add this repository as a package dependency to your Swift project](https://developer.apple.com/documentation/xcode/adding-package-dependencies-to-your-app#Add-a-package-dependency). After adding the package, you'll need to set up SQLite with extension loading by following steps 4 and 5 of [this guide](https://github.com/sqliteai/sqlite-extensions-guide/blob/main/platforms/ios.md#4-set-up-sqlite-with-extension-loading).

Here's an example of how to use the package:
```swift
import vector

...

var db: OpaquePointer?
sqlite3_open(":memory:", &db)
sqlite3_enable_load_extension(db, 1)
var errMsg: UnsafeMutablePointer<Int8>? = nil
sqlite3_load_extension(db, vector.path, nil, &errMsg)
var stmt: OpaquePointer?
sqlite3_prepare_v2(db, "SELECT vector_version()", -1, &stmt, nil)
defer { sqlite3_finalize(stmt) }
sqlite3_step(stmt)
log("vector_version(): \(String(cString: sqlite3_column_text(stmt, 0)))")
sqlite3_close(db)
```

### Android Package

Add the [following](https://central.sonatype.com/artifact/ai.sqlite/vector) to your Gradle dependencies:

```gradle
implementation 'ai.sqlite:vector:0.9.80'
```

Here's an example of how to use the package:
```java
SQLiteCustomExtension vectorExtension = new SQLiteCustomExtension(getApplicationInfo().nativeLibraryDir + "/vector", null);
SQLiteDatabaseConfiguration config = new SQLiteDatabaseConfiguration(
    getCacheDir().getPath() + "/vector_test.db",
    SQLiteDatabase.CREATE_IF_NECESSARY | SQLiteDatabase.OPEN_READWRITE,
    Collections.emptyList(),
    Collections.emptyList(),
    Collections.singletonList(vectorExtension)
);
SQLiteDatabase db = SQLiteDatabase.openDatabase(config, null, null);
```

**Note:** Additional settings and configuration are required for a complete setup. For full implementation details, see the [complete Android example](https://github.com/sqliteai/sqlite-extensions-guide/blob/main/examples/android/README.md).

### Python Package

Python developers can quickly get started using the ready-to-use `sqlite-vector` package available on PyPI:

```bash
pip install sqliteai-vector
```

For usage details and examples, see the [Python package documentation](./packages/python/README.md).

### Flutter Package

Add the [sqlite_vector](https://pub.dev/packages/sqlite_vector) package to your project:

```bash
flutter pub add sqlite_vector  # Flutter projects
dart pub add sqlite_vector     # Dart projects
```

Usage with `sqlite3` package:
```dart
import 'package:sqlite3/sqlite3.dart';
import 'package:sqlite_vector/sqlite_vector.dart';

sqlite3.loadSqliteVectorExtension();
final db = sqlite3.openInMemory();
print(db.select('SELECT vector_version()'));
```

For a complete example, see the [Flutter example](https://github.com/sqliteai/sqlite-extensions-guide/blob/main/examples/flutter/README.md).

## Documentation

Extensive API documentation can be found in the [API page](https://github.com/sqliteai/sqlite-vector/blob/main/API.md).

More information about the quantization process can be found in the [QUANTIZATION document](https://github.com/sqliteai/sqlite-vector/blob/main/QUANTIZATION.md).

## Features

### Instant Vector Search – No Preindexing Required

Unlike other SQLite vector extensions that rely on complex indexing algorithms such as DiskANN, HNSW, or IVF, which often require **preprocessing steps that can take hours or even days**, `sqlite-vector` works out of the box with your existing data. There’s **no need to preindex your vectors**—you can start performing fast, approximate or exact vector searches **immediately**.

This means:

* **No waiting time** before your app or service is usable
* **Zero-cost updates** – you can add, remove, or modify vectors on the fly without rebuilding any index
* **Works directly with BLOB columns** in ordinary SQLite tables – no special schema or virtual table required
* **Ideal for edge and mobile use cases**, where preprocessing large datasets is not practical or possible

By eliminating the need for heavyweight indexing, `sqlite-vector` offers a **simpler, faster, and more developer-friendly** approach to embedding vector search in your applications.

### Supported Vector Types

You can store your vectors as `BLOB` columns in ordinary tables. Supported formats include:

* `float32` (4 bytes per element)
* `float16` (2 bytes per element)
* `bfloat16` (2 bytes per element)
* `int8` (1 byte per element)
* `uint8` (1 byte per element)
* `1bit` (1 bit per element)

Simply insert a vector as a binary blob into your table. No special table types or schemas are required.

A stored column is quantized separately with `vector_quantize(table, column, 'qtype=...')`,
which builds a compact index the scan reads instead of the raw vectors:

| `qtype` | Bytes per dimension | Notes |
| --- | ---: | --- |
| `UINT8` | 1 | Asymmetric. Correct for L2; see the [benchmark](#benchmark) before using it with cosine |
| `INT8` | 1 | Symmetric. The default choice for cosine |
| `1BIT` | 1/8 | Hamming only. A pre-filter to re-rank, not a final ranking |
| `TURBO2` / `TURBO3` / `TURBO4` | 1/4, 3/8, 1/2 | Lookup-table scan; smallest indexes, see [TurboQuant](#turboquant-benchmark-and-recall) |

Omitting `qtype` picks `UINT8` for non-negative data and `INT8` otherwise. `BIT` columns
are already binary, so `1BIT` is the only quantization they accept.


### Supported Distance Metrics

Optimized implementations available:

* **L2 Distance (Euclidean)**
* **Squared L2**
* **L1 Distance (Manhattan)**
* **Cosine Distance**
* **Dot Product**
* **Hamming Distance** (available only with 1bit vectors — `vector_init()` rejects it for any other type)

These are implemented in pure C and optimized for SIMD when available, ensuring maximum performance on modern CPUs and mobile devices.

If your embeddings are already unit length, say so with `normalized=1`: cosine on a
`FLOAT32` column then reduces to `1 - dot`, dropping two thirds of the arithmetic from the
inner loop for the same results. It is an assertion about your data, not a request — see
[API.md](API.md#vector_inittable-column-options).

---

# What Is Vector Search?

Vector search is the process of finding the closest match(es) to a given vector (a point in high-dimensional space) based on a similarity or distance metric. It is essential for AI and machine learning applications where data is often encoded into vector embeddings.

### Common Use Cases

* **Semantic Search**: find documents, emails, or messages similar to a query
* **Image Retrieval**: search for visually similar images
* **Recommendation Systems**: match users with products, videos, or music
* **Voice and Audio Search**: match voice queries or environmental sounds
* **Anomaly Detection**: find outliers in real-time sensor data
* **Robotics**: localize spatial features or behaviors using embedded observations

In the AI era, embeddings are everywhere – from language models like GPT to vision transformers. Storing and searching them efficiently is the foundation of intelligent applications.

## Perfect for Edge AI

SQLite-Vector is designed with the **Edge AI** use case in mind:

* Runs offline – no internet required
* Works on mobile devices – iOS/Android friendly
* Keeps data local – ideal for privacy-focused apps
* Extremely fast – real-time performance on device

You can deploy powerful similarity search capabilities right inside your app or embedded system – **no cloud needed**.

---

## License

Free Use in Open-Source Projects: You may use, copy, distribute, and prepare derivative works of the software — in source or object form, with or without modification — freely and without fee, provided the software is incorporated into or used by an open-source project licensed under an OSI-approved open-source license. Everything else is licensed under the [Elastic License 2.0](./LICENSE.md). You can use, copy, modify, and distribute it under the terms of the license for non-production use. For production or managed service use, please [contact SQLite Cloud, Inc](mailto:info@sqlitecloud.io) for a commercial license.

---


## ☁️ Hosted version

Don't want to run it yourself? **[SQLite Cloud](https://sqlite.ai)** is the managed version of SQLite-Vector and the rest of the stack — with sync, backups, auth, edge functions, and multi-region support included.

[**Start free →**](https://dashboard.sqlitecloud.io/auth/sign-in)

---

## Part of the SQLite AI stack

SQLite-Vector is one piece of a larger ecosystem that turns SQLite into a runtime for intelligent, distributed data:

**Data layer**
- [**sqlite-vector**](https://github.com/sqliteai/sqlite-vector) — ANN vector search inside SQLite *(you are here)*
- [sqlite-sync](https://github.com/sqliteai/sqlite-sync) — Offline-first CRDT sync across devices
- [sqlite-columnar](https://github.com/sqliteai/sqlite-columnar) — Column-oriented analytics for OLAP queries
- [sqlite-js](https://github.com/sqliteai/sqlite-js) — Custom SQLite functions written in JavaScript

**AI layer**
- [sqlite-ai](https://github.com/sqliteai/sqlite-ai) — On-device LLM inference and embeddings
- [sqlite-agent](https://github.com/sqliteai/sqlite-agent) — Autonomous AI agents running inside SQLite
- [sqlite-memory](https://github.com/sqliteai/sqlite-memory) — Persistent, searchable memory for agents
- [sqlite-mcp](https://github.com/sqliteai/sqlite-mcp) — Call MCP tools directly from SQL queries

**Managed platform**
- [SQLite Cloud](https://sqlite.ai) — Hosted SQLite with sync, auth, edge functions, and analytics. [Free tier →](https://dashboard.sqlitecloud.io/auth/sign-in)

Built by [SQLite AI](https://sqlite.ai). Questions? [Contact us](https://sqlite.ai/support).

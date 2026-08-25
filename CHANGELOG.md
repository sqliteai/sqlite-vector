# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

Documentation and tooling only — the extension itself is unchanged from 1.1.0, so a build
from this revision produces the same binaries.

### Changed

- **`make benchmark` now uses a file-backed database, never `:memory:`.** An in-memory
  database keeps the whole index in the process however the scan is configured, which made
  the memory figures meaningless: the point of a streamed scan is that the index is *not*
  resident. Reported figures move accordingly — the exact `FLOAT32` scan goes from 148 to
  484 ms/query because it now pays to read 3 GB, and a streamed `INT8` scan from 56 to
  114 ms. A preloaded scan is essentially unchanged at 37.6 ms, because after the one-time
  load it reads the extension's own buffer and never returns to SQLite.
- **The README benchmark table now compares hardware rather than modes**, two rows per
  machine, with the peak memory measured rather than the configured limit read back.
  `make benchmark HARDWARE="<your CPU>"` prints those rows ready to paste; it appends the
  backend the build actually selected rather than trusting a hand-written label, and
  refuses to print rows at all when the run's parameters differ from the ones the table is
  built on.

### Fixed

- **Corrected a claim in the README**: TurboQuant was described as slower than the exact
  scan, which was an artifact of measuring in memory. File-backed, `TURBO4` is **3.2x
  faster** than the exact scan — it does beat brute force. What holds is the comparison
  that mattered: `INT8` is 4x faster again at twice the index size, so TurboQuant's
  argument is memory rather than throughput.

## [1.1.0] - 2026-08-24

A source audit closed thirteen defects, including two crashes and four memory-safety
issues, and found that every x86 release had been shipping scalar code. Search is
substantially faster as a result, and the SIMD kernels are now actually tested.

### Added

- **`normalized=1` option for `vector_init()`**: declares that every stored vector is unit length. With `type=FLOAT32` and `distance=COSINE` a full-precision scan then computes `1 - dot` instead of the full cosine, dropping two thirds of the arithmetic from the inner loop. The query is normalized once per scan, so reported distances are unchanged. It is an assertion about your data, not a request: if the stored vectors are not unit length the distances will be wrong. Quantized scans ignore it.
- **`make benchmark`**: brute-force k-NN benchmark across every storage and quantization mode, with recall scored against the exact scan. Defaults to k=20 over 1,000,000 vectors of dimension 768; see the Benchmark section of the README.
- **`make unittest-simd`**: runs the test suite against the SIMD kernels. The existing `unittest` target builds every source in one invocation, which left `__AVX2__` and `__AVX512F__` undefined and silently exercised the scalar fallback instead.
- **CI coverage for the AVX-512 kernels**, on hardware where the runner has it and under Intel SDE where it does not, with an assertion that the expected backend was the one that ran.

### Changed

- **x86 builds now contain the AVX2 and AVX-512 kernels.** They were guarded by `__AVX2__` / `__AVX512F__`, which the build never defined, so both compiled to nothing — and because the runtime dispatch was an `if / else if` ladder, a CPU reporting AVX2 called an empty stub and never fell back to the SSE2 kernels that were compiled in. Every x86 release so far ran the plain C fallback.
- **Faster distance kernels.** The FLOAT32 kernels now use four independent accumulators and true FMA; the UINT8/INT8 kernels were rewritten around the instructions built for them (`vabd`/`vmull`/`vpadal` on NEON, `PSADBW`/`PMADDWD` on x86). AVX2 and AVX-512 cosine no longer makes three separate passes over the data. Measured on Apple M5 Pro at dimension 768, a preloaded quantized scan went from 17.4 to 62.6 Mvec/s. Accuracy improved as well: reductions widen to 64 bits before folding lanes, and integer cosine sums exact integers rather than accumulating in `float`.
- **Faster top-k.** The candidate set is a binary max-heap instead of a linear rescan plus an exchange sort. At k=1000 over 20,000 rows a 1-bit scan went from 3.17 ms to 0.16 ms per query; at k=4000, from 33.5 ms to 0.55 ms. Small k is unchanged.
- **TurboQuant lookup scans return the same distance on every CPU.** The per-backend implementations accumulated in `float` while the scalar one used `double`, so results differed by up to 1.5e-4 relative depending on which backend ran — enough to reorder near-ties. There is now one implementation.
- **`vector_turboquant_backend()`** still returns the same strings, but the value identifies the SIMD tier selected at load time rather than a TurboQuant-specific code path, since there is only one.
- **`qtype=AUTO` on a `BIT` column now means `1BIT`.** Previously it failed on a populated table with an unrelated message, and on an empty one silently recorded `UINT8`, which then applied to rows inserted later. An explicit 8-bit request on a `BIT` column is now refused with a message that says so.
- **`distance=HAMMING` is rejected for any type other than `BIT`** at `vector_init()` time.

### Fixed

- **Crash on `distance=HAMMING` with a non-`BIT` vector type.** The dispatch table only implements Hamming for `BIT`, and the combination was accepted, so scanning called a NULL function pointer.
- **Out-of-bounds read from an undersized query vector.** A query passed as a `BLOB` was never length-checked, so the distance kernels read `dimension` elements from whatever the caller supplied. The JSON form already validated this.
- **SQL injection through table and column names.** Identifiers were interpolated with `%q`, which escapes string literals and does nothing for `;` or `"`. A table whose name contains a semicolon could inject statements through `vector_quantize_cleanup()`. This also fixes ordinary names: tables or columns containing spaces or dashes previously failed with a syntax error and now work.
- **`ORDER BY` was silently ignored** on `vector_full_scan()` and `vector_quantize_scan()`. The planner was told the cursor already satisfied any ordering, so SQLite dropped the sorter — including for `ORDER BY distance DESC`.
- **Out-of-bounds reads on a malformed quantized index.** The `UINT8`/`INT8`/`1BIT` scan paths decoded rows without checking the blob against the row count it claimed. The shadow table is an ordinary writable table.
- **Heap buffer overflow in `vector_quantize_preload()`.** The buffer was sized from `SUM(LENGTH(data))` and filled with no per-row bound. No concurrency was needed to trigger it: `LENGTH()` counts characters on a TEXT value while the byte length can be larger.
- **Use-after-free when `vector_quantize_cleanup()` or `vector_quantize_preload()` ran while a streaming cursor was open.** The in-memory index is now reference counted, so a scan keeps reading a consistent snapshot.
- **Double free on the extension-init error path**, which SQLite could reach immediately because it invokes the destructor when `sqlite3_create_function_v2()` itself fails.
- **`k = 0`** returned an error code from `xFilter` instead of an empty result.
- **Undefined float-to-int conversion** in the unrolled 8-bit quantizers for NaN and out-of-range values.
- **Primary-key detection on `WITHOUT ROWID` tables** bound a parameter to a statement that has none and read columns from an arbitrary row of a grouped query.
- **`vector_quantize()` reported "not an error"** whenever the failure came from the extension rather than from SQLite.
- **Uninitialised bytes** in the index when a `BIT` column was quantized to 8 bits.
- **GCC 13 build failure on AVX2 targets**: a static `__m256i` initializer is now built from a plain byte array, so the extension compiles with GCC 13's stricter constant-expression rules.
- **Swift Package**: removed the deprecated `.iOS(.v11)` platform declaration that produced a warning (and, on recent toolchains, an error) when resolving the package.

### Notes

- **For cosine, prefer `qtype=INT8` over `UINT8`.** They are the same size and the same speed, but unsigned quantization subtracts the dataset minimum before scaling, and cosine measures angle, which that shift destroys. Omitting `qtype` selects `UINT8` for non-negative data, which is correct for L2 and wrong for cosine. See the Benchmark section of the README.
- **Tie-breaking among equal distances changed** with the new top-k. Neither ordering was stable and none was guaranteed, but it is observable.

## [1.0.0] - 2026-05-25

### Added

- **Google TurboQuant support**: a new family of quantization types (`qtype=TURBO`, `TURBO2`, `TURBO3`, `TURBO4`, plus the `qbits` option for `2`/`3`/`4` bit widths) for `vector_quantize()`. TurboQuant uses lookup-table based scanning with SIMD acceleration on SSE2, NEON and RVV, giving substantially faster approximate search at a small recall cost compared to the previous quantization modes.
- **`vector_turboquant_backend()`**: returns the SIMD backend actually selected for TurboQuant lookup-table scans, so you can verify the expected code path is used on a target runtime.

### Fixed

- Linux MUSL test failures.
- `succesfully` → `successfully` typo in `API.md`.

## [0.9.95] - 2026-04-07

### Fixed

- **CI/CD**: the Flutter package publish workflow now triggers correctly on new releases.

## [0.9.94] - 2026-04-01

### Fixed

- **Swift Package**: use a binary target and a versioned macOS framework for Xcode 26 compatibility.

## [0.9.93] - 2026-03-11

### Fixed

- **Flutter**: iOS simulator native asset packaging.
- **CI/CD**: Python and Flutter package publish workflows.

## [0.9.92] - 2026-02-21

### Added

- **RISC-V Vector (RVV) distance functions for all supported vector types**: `f32`, `f16`, `bf16`, `i8`, `u8` and binary (Hamming), covering the full set of distance metrics. RISC-V hosts with RVV now get SIMD acceleration instead of falling back to the generic CPU path.

## [0.9.91] - 2026-02-21

### Added

- **RISC-V 64 support**: RVV feature detection at runtime, cosine distance implemented with RVV intrinsics, and the target architecture string passed through to the compiler.

### Fixed

- **Streaming scans ignored `ORDER BY` (#43)**: `vector_full_scan_stream()` / `vector_quantize_scan_stream()` incorrectly reported to SQLite that they already returned rows in the requested order, so SQLite skipped sorting entirely. Queries combining streaming mode with `ORDER BY` (and `JOIN` / `LIMIT`) returned rows in scan order rather than by distance. Top-k mode is now detected correctly and streaming results are sorted by SQLite as expected.

## [0.9.90] - 2026-02-16

### Fixed

- **Windows**: `libgcc` is now statically linked, so the extension no longer fails to load because of missing runtime DLL dependencies (#42).

## [0.9.85] - 2026-02-11

### Fixed

- **Flutter package**: corrected the example path and removed the obsolete example.

## [0.9.84] - 2026-02-11

### Added

- **Flutter multi-platform package**, published to pub.dev, with install instructions in the README.

## [0.9.80] - 2026-02-09

### Changed

- **Unified streaming and non-streaming scan implementations**: `vector_full_scan()` / `vector_quantize_scan()` and their `_stream` variants now share a single code path, removing behavioral differences between them and simplifying the documented API surface.
- **Distance results are now clamped** to their valid ranges across all backends (CPU, SSE2, AVX2, AVX512, NEON), preventing tiny floating-point overshoots (e.g. a cosine distance marginally below `0` or above `2`).

## [0.9.70] - 2026-01-26

### Fixed

- **Android**: increased the maximum supported page size to 16 KB, fixing loading on devices and emulators using 16 KB memory pages (#36).

## [0.9.60] - 2026-01-22

### Added

- **Binary (1-bit) vectors**: new `BIT` vector type with `vector_as_bit()` conversion and `qtype=BIT` quantization, plus SIMD-optimized Hamming distance on SSE2, AVX2, AVX512 and NEON.
- **AVX512 distance backend** for x86 CPUs that support it, selected automatically ahead of AVX2.

### Fixed

- Several minor memory leaks and code-consistency issues.

## [0.9.52] - 2025-11-05

### Added

- **Android `armeabi-v7a` (32-bit ARM NEON) support** (#30).

## [0.9.51] - 2025-10-20

### Fixed

- **npm package**: removed `package-lock.json` from the published files and cleaned the ignore lists, fixing integrity and resolution errors on install.

## [0.9.50] - 2025-10-20

### Fixed

- **npm package**: corrected the `LICENSE` path referenced by the platform-specific package READMEs.

## [0.9.49] - 2025-10-17

### Changed

- Release notes now list all supported packages.

## [0.9.47] - 2025-10-17

### Fixed

- **npm package**: updated the package README and switched publishing to OIDC authentication.

## [0.9.46] - 2025-10-16

### Fixed

- **npm package**: automatic package version updates on release.

## [0.9.45] - 2025-10-16

### Added

- **`@sqliteai/sqlite-vector` npm package**, with platform-specific packages published for each supported OS/architecture.

## [0.9.39] - 2025-10-16

### Fixed

- **WASM**: the `sqlite-wasm` version is now bumped in `package.json` on release.

## [0.9.38] - 2025-10-15

### Added

- **Streaming scan interface**: `vector_full_scan_stream()` and `vector_quantize_scan_stream()` are now documented in `API.md`, including how to access additional columns from the scanned table.

## [0.9.37] - 2025-10-10

### Fixed

- **WASM compilation** failures.
- Added stricter argument validation to several critical functions.

## [0.9.35] - 2025-10-06

### Added

- **Separate per-architecture macOS builds** in releases (in addition to the universal build).

### Fixed

- **Apple builds**: added `-headerpad_max_install_names` so install names can be rewritten during app packaging and code signing.

## [0.9.34] - 2025-10-03

### Fixed

- **Release**: skip the duplicate Android AAR build.

## [0.9.33] - 2025-10-02

### Added

- Android package integration instructions in the README.

## [0.9.32] - 2025-10-02

### Added

- **Android AAR published to Maven Central**.

## [0.9.28] - 2025-10-01

### Added

- **JitPack configuration** for building the Android AAR package.

## [0.9.27] - 2025-09-29

### Added

- **Android AAR package** build support.

## [0.9.25] - 2025-09-29

### Fixed

- **Apple releases**: the `.dylib` extensions are now notarized, so macOS no longer blocks them on first load.

## [0.9.24] - 2025-09-17

### Added

- **Swift Package support**, so the extension can be consumed directly from Xcode projects.

## [0.9.23] - 2025-09-09

### Fixed

- A possible failure during quantization.

## [0.9.22] - 2025-09-08

### Added

- **WebAssembly (WASM) support**, with a `sqlite-wasm` build published alongside the native releases and a download link documented in the README.

## [0.9.21] - 2025-08-29

### Added

- **Linux MUSL builds**, **Apple XCFramework builds**, and signing/notarization for Apple artifacts.

### Fixed

- `strcasestr` definition for compatibility with Linux MUSL.

## [0.9.20] - 2025-08-29

### Added

- **Preliminary streaming scans**: `vector_full_scan_stream()` and `vector_quantize_scan_stream()`, which return results incrementally instead of materializing the full top-k result set.

## [0.9.11] - 2025-08-28

### Changed

- **`vector_cleanup()` renamed to `vector_quantized_cleanup()`** (later `vector_quantize_cleanup()`), to make it clear it removes quantization data rather than all extension state.

### Added

- Python package build workflow (`sqlite-vector` on PyPI).

## [0.9.9] - 2025-08-25

### Fixed

- **NULL vectors are now correctly skipped during scanning** instead of producing bogus distances.
- Improved rounding of values close to zero during quantization.

## [0.9.8] - 2025-08-23

### Fixed

- Several issues affecting quantization of non-`FLOAT32` vector types.
- An incomplete error message.

### Changed

- **License**: `LICENSE.md` now explicitly grants usage for open-source projects.

## [0.9.6] - 2025-08-22

### Added

- **`float16` and `bfloat16` vector types**.

### Changed

- **`vector_convert_type()` renamed to `vector_as_type()`** (`vector_as_f32()`, `vector_as_f16()`, `vector_as_bf16()`, `vector_as_i8()`, `vector_as_u8()`).
- Removed the check that rejected unsupported types.

### Fixed

- Android builds now link the math library.

## [0.9.4] - 2025-08-21

### Fixed

- **`int8` cosine distance** now accumulates in `int32_t` internally instead of floats, improving accuracy and speed.

## [0.9.1] - 2025-08-04

### Changed

- **Better error message** when `vector_quantize_scan()` is used without first calling `vector_quantize()`.

## [0.8.9] - 2025-08-04

### Added

- **Signed 8-bit (`INT8`) quantization** (`qtype=INT8`).

## [0.8.8] - 2025-07-21

### Fixed

- The extension context is now correctly deallocated, fixing memory leaks on connection close.

## [0.8.7] - 2025-07-02

### Added

- **Optional `dimension` argument** for the `vector_convert_*` functions.
- **`API.md`**, **`QUANTIZATION.md`** and **`LICENSE.md`** documentation.
- A semantic search example.

## [0.8.6] - 2025-06-24

### Added

- **All `vector_convert_*` functions are now exposed** as SQL functions.
- **Windows support**, including a local `strcasestr` implementation.

### Fixed

- Cross-platform float min/max initialization.
- Android build and test setup.

## [0.8.5] - 2025-06-23

### Added

- First public release of the SQLite Vector extension: vector storage and search over ordinary SQLite tables.
- **`vector_init()`**, **`vector_quantize()`**, **`vector_full_scan()`**, **`vector_quantize_scan()`**, **`vector_cleanup()`**, **`vector_version()`** and **`vector_backend()`**.
- **Cross-platform distance functions** with a runtime dynamic dispatch table selecting the best available backend (CPU, SSE2, AVX2, NEON).

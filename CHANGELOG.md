# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Fixed

- **GCC 13 build failure on AVX2 targets**: a static `__m256i` initializer is now built from a plain byte array, so the extension compiles with GCC 13's stricter constant-expression rules.
- **Swift Package**: removed the deprecated `.iOS(.v11)` platform declaration that produced a warning (and, on recent toolchains, an error) when resolving the package.

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

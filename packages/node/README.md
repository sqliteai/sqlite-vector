# @sqliteai/sqlite-vector

[![npm version](https://badge.fury.io/js/@sqliteai%2Fsqlite-vector.svg)](https://badge.fury.io/js/@sqliteai%2Fsqlite-vector)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE.md)

> SQLite Vector extension packaged for Node.js

**SQLite Vector** is a cross-platform, ultra-efficient SQLite extension that brings vector search capabilities to your embedded database. It works seamlessly on **iOS, Android, Windows, Linux, and macOS**, using just **30MB of memory** by default. With support for **Float32, Float16, BFloat16, Int8, and UInt8**, and **highly optimized distance functions**, it's the ideal solution for **Edge AI** applications.

## Features

- ✅ **Cross-platform** - Works on macOS, Linux (glibc/musl), and Windows
- ✅ **Zero configuration** - Automatically detects and loads the correct binary for your platform
- ✅ **TypeScript native** - Full type definitions included
- ✅ **Modern ESM + CJS** - Works with both ES modules and CommonJS
- ✅ **Small footprint** - Only downloads binaries for your platform
- ✅ **Offline-ready** - No external services required

## Installation

```bash
npm install @sqliteai/sqlite-vector
```

The package automatically downloads the correct native extension for your platform during installation.

### Supported Platforms

| Platform | Architecture | Package |
|----------|-------------|---------|
| macOS | ARM64 (Apple Silicon) | `@sqliteai/sqlite-vector-darwin-arm64` |
| macOS | x86_64 (Intel) | `@sqliteai/sqlite-vector-darwin-x86_64` |
| Linux | ARM64 (glibc) | `@sqliteai/sqlite-vector-linux-arm64` |
| Linux | ARM64 (musl/Alpine) | `@sqliteai/sqlite-vector-linux-arm64-musl` |
| Linux | x86_64 (glibc) | `@sqliteai/sqlite-vector-linux-x86_64` |
| Linux | x86_64 (musl/Alpine) | `@sqliteai/sqlite-vector-linux-x86_64-musl` |
| Windows | x86_64 | `@sqliteai/sqlite-vector-win32-x86_64` |

## sqlite-vector API

For detailed information on how to use the vector extension features, see the [main documentation](https://github.com/sqliteai/sqlite-vector/blob/main/README.md).

## Usage

```typescript
import { getExtensionPath } from '@sqliteai/sqlite-vector';
import Database from 'better-sqlite3';

const db = new Database(':memory:');
db.loadExtension(getExtensionPath());

// Ready to use
const version = db.prepare('SELECT vector_version()').pluck().get();
console.log('Vector extension version:', version);
```

## Examples

For complete, runnable examples, see the [sqlite-extensions-guide](https://github.com/sqliteai/sqlite-extensions-guide/tree/main/examples/node).

These examples are generic and work with all SQLite extensions: `sqlite-vector`, `sqlite-sync`, `sqlite-js`, and `sqlite-ai`.

## API Reference

### `getExtensionPath(): string`

Returns the absolute path to the SQLite Vector extension binary for the current platform.

**Returns:** `string` - Absolute path to the extension file (`.so`, `.dylib`, or `.dll`)

**Throws:** `ExtensionNotFoundError` - If the extension binary cannot be found for the current platform

**Example:**
```typescript
import { getExtensionPath } from '@sqliteai/sqlite-vector';

const path = getExtensionPath();
// => '/path/to/node_modules/@sqliteai/sqlite-vector-darwin-arm64/vector.dylib'
```

---

### `getExtensionInfo(): ExtensionInfo`

Returns detailed information about the extension for the current platform.

**Returns:** `ExtensionInfo` object with the following properties:
- `platform: Platform` - Current platform identifier (e.g., `'darwin-arm64'`)
- `packageName: string` - Name of the platform-specific npm package
- `binaryName: string` - Filename of the binary (e.g., `'vector.dylib'`)
- `path: string` - Full path to the extension binary

**Throws:** `ExtensionNotFoundError` - If the extension binary cannot be found

**Example:**
```typescript
import { getExtensionInfo } from '@sqliteai/sqlite-vector';

const info = getExtensionInfo();
console.log(`Running on ${info.platform}`);
console.log(`Extension path: ${info.path}`);
```

---

### `getCurrentPlatform(): Platform`

Returns the current platform identifier.

**Returns:** `Platform` - One of:
- `'darwin-arm64'` - macOS ARM64
- `'darwin-x86_64'` - macOS x86_64
- `'linux-arm64'` - Linux ARM64 (glibc)
- `'linux-arm64-musl'` - Linux ARM64 (musl)
- `'linux-x86_64'` - Linux x86_64 (glibc)
- `'linux-x86_64-musl'` - Linux x86_64 (musl)
- `'win32-x86_64'` - Windows x86_64

**Throws:** `Error` - If the platform is unsupported

---

### `isMusl(): boolean`

Detects if the system uses musl libc (Alpine Linux, etc.).

**Returns:** `boolean` - `true` if musl is detected, `false` otherwise

---

### `class ExtensionNotFoundError extends Error`

Error thrown when the SQLite Vector extension cannot be found for the current platform.

## Related Projects

- **[@sqliteai/sqlite-ai](https://www.npmjs.com/package/@sqliteai/sqlite-ai)** - On-device AI inference and embedding generation
- **[@sqliteai/sqlite-sync](https://www.npmjs.com/package/@sqliteai/sqlite-sync)** - Sync on-device databases with the cloud
- **[@sqliteai/sqlite-js](https://www.npmjs.com/package/@sqliteai/sqlite-js)** - Define SQLite functions in JavaScript

## License

This project is licensed under the [Apache License 2.0](LICENSE.md).

## Contributing

Contributions are welcome! Please see the [main repository](https://github.com/sqliteai/sqlite-vector) to open an issue.

## Support

- 📖 [Documentation](https://github.com/sqliteai/sqlite-vector/blob/main/API.md)
- 🐛 [Report Issues](https://github.com/sqliteai/sqlite-vector/issues)

## SQLite Vector Python package

This package provides the sqlite-vector extension prebuilt binaries for multiple platforms and architectures.

### SQLite Vector

SQLite Vector is a cross-platform, ultra-efficient SQLite extension that brings vector search capabilities to your embedded database. It works seamlessly on iOS, Android, Windows, Linux, and macOS, using just 30MB of memory by default. With support for Float32, Float16, BFloat16, Int8, and UInt8, and highly optimized distance functions, it's the ideal solution for Edge AI applications.

More details on the official repository [sqliteai/sqlite-vector](https://github.com/sqliteai/sqlite-vector).

### Documentation

For detailed information on all available functions, their parameters, and examples, refer to the [comprehensive API Reference](https://github.com/sqliteai/sqlite-vector/blob/main/API.md).

### Supported Platforms and Architectures

| Platform      | Arch         | Subpackage name          | Binary name  |
| ------------- | ------------ | ------------------------ | ------------ |
| Linux (CPU)   | x86_64/arm64 | sqlite-vector.binaries   | vector.so    |
| Windows (CPU) | x86_64       | sqlite-vector.binaries   | vector.dll   |
| macOS (CPU)   | x86_64/arm64 | sqlite-vector.binaries   | vector.dylib |

## Usage

> **Note:** Some SQLite installations on certain operating systems may have extension loading disabled by default.   
If you encounter issues loading the extension, refer to the [sqlite-extensions-guide](https://github.com/sqliteai/sqlite-extensions-guide/) for platform-specific instructions on enabling and using SQLite extensions.

```python
import importlib.resources
import sqlite3

# Connect to your SQLite database
conn = sqlite3.connect("example.db")

# Load the sqlite-vector extension
# pip will install the correct binary package for your platform and architecture
ext_path = importlib.resources.files("sqlite-vector.binaries") / "vector"

conn.enable_load_extension(True)
conn.load_extension(str(ext_path))
conn.enable_load_extension(False)


# Now you can use sqlite-vector features in your SQL queries
print(conn.execute("SELECT vector_version();").fetchone())
```
# TRX-cpp

TRX-cpp is a C++11 library for reading, writing, and memory-mapping the TRX
tractography file format (zip archives or on-disk directories of memmaps).

## Dependencies

Required:
- C++11 compiler and CMake (>= 3.10)
- libzip
- nlohmann::json
- Eigen3
- spdlog

Optional:
- GTest (for building tests)

## Build and Install

### Quick build (no tests)

```
cmake -S . -B build
cmake --build build
```

### Build with tests

```
cmake -S . -B build \
  -DTRX_BUILD_TESTS=ON \
  -DGTest_DIR=/path/to/GTestConfig.cmake
cmake --build build
ctest --test-dir build --output-on-failure
```

### mio include path

`mio` is vendored under `third_party/mio` and headers are copied into
`include/mio` for downstream builds. If you want to use a system-provided `mio`,
point CMake at it:

```
cmake -S . -B build -DMIO_INCLUDE_DIR=/path/to/mio/include
```

## Third-party notices

- `mio` by Martin Andreyel (https://github.com/mandreyel/mio) is vendored in
  `third_party/mio`. See `third_party/mio/LICENSE` for the license text.

### Filesystem shim

The project relies on the built-in lightweight filesystem shim and does not
depend on Boost.

## Usage Examples

All examples assume:

```
#include <trx/trx.h>

using namespace trxmmap;
```

### Read a TRX zip and inspect data

```
TrxFile<half> *trx = load_from_zip<half>("tracks.trx");

// Access streamlines: vertices are stored as an Eigen matrix
const auto num_vertices = trx->streamlines->_data.size() / 3;
const auto num_streamlines = trx->streamlines->_offsets.size() - 1;

std::cout << "Vertices: " << num_vertices << "\n";
std::cout << "Streamlines: " << num_streamlines << "\n";
std::cout << "First vertex (x,y,z): "
          << trx->streamlines->_data(0, 0) << ","
          << trx->streamlines->_data(0, 1) << ","
          << trx->streamlines->_data(0, 2) << "\n";

// Data-per-streamline and data-per-vertex are stored in maps
for (const auto &kv : trx->data_per_streamline) {
  std::cout << "DPS '" << kv.first << "' elements: "
            << kv.second->_matrix.size() << "\n";
}
for (const auto &kv : trx->data_per_vertex) {
  std::cout << "DPV '" << kv.first << "' elements: "
            << kv.second->_data.size() << "\n";
}

trx->close(); // cleans up temporary on-disk data
delete trx;
```

### Read from an on-disk TRX directory

```
TrxFile<float> *trx = load_from_directory<float>("/path/to/trx_dir");
std::cout << "Header JSON:\n" << trx->header.dump(2) << "\n";
trx->close();
delete trx;
```

### Write a TRX file

You can modify a loaded `TrxFile` and save it to a new archive:

```
TrxFile<half> *trx = load_from_zip<half>("tracks.trx");

// Example: update header metadata
trx->header["COMMENT"] = "saved by trx-cpp";

// Save with no compression (ZIP_CM_STORE) or another libzip compression level
save(*trx, "tracks_copy.trx", ZIP_CM_STORE);

trx->close();
delete trx;
```

### Notes on memory mapping

`TrxFile` uses memory-mapped arrays under the hood for large datasets. The
`close()` method cleans up any temporary folders created during zip extraction.
If you skip `close()`, temporary directories may be left behind.

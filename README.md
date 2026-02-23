# TRX-cpp

[![Documentation](https://readthedocs.org/projects/trx-cpp/badge/?version=latest)](https://trx-cpp.readthedocs.io/en/latest/)
[![codecov](https://codecov.io/gh/tee-ar-ex/trx-cpp/branch/main/graph/badge.svg)](https://codecov.io/gh/tee-ar-ex/trx-cpp)

A C++17 library for reading, writing, and memory-mapping the [TRX tractography format](https://github.com/tee-ar-ex/trx-spec) — efficient storage for large-scale tractography data.

## Features

- **Zero-copy memory mapping** — positions, DPV, and DPS arrays are exposed as `Eigen::Map` views directly over memory-mapped files; no unnecessary copies for large tractograms
- **Streaming writes** — `TrxStream` appends streamlines one at a time and finalizes to a standard TRX archive or directory, suitable when the total count is unknown at the start
- **Spatial queries** — build per-streamline axis-aligned bounding boxes (AABBs) and efficiently extract spatial subsets; designed for interactive slice-view workflows
- **Typed and type-erased APIs** — `TrxFile<DT>` gives compile-time type safety; `AnyTrxFile` dispatches at runtime when the dtype is read from disk
- **ZIP and directory storage** — read and write `.trx` zip archives and plain on-disk directories with the same API
- **Optional NIfTI support** — read qform/sform affines from `.nii` / `.nii.gz` and embed them in the TRX header

## Quick start

**Dependencies:** a C++17 compiler, [libzip](https://libzip.org/), [Eigen 3.4+](https://eigen.tuxfamily.org/)

```cmake
# CMakeLists.txt
find_package(trx-cpp CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE trx-cpp::trx)
```

```cpp
#include <trx/trx.h>

// Load any TRX file — dtype detected at runtime
auto trx = trx::load_any("tracks.trx");

std::cout << trx.num_streamlines() << " streamlines, "
          << trx.num_vertices()    << " vertices\n";

// Access positions as an Eigen matrix (zero-copy)
auto positions = trx.positions.as_matrix<float>(); // (NB_VERTICES, 3)

trx.close();
```

See [Building](https://trx-cpp.readthedocs.io/en/latest/building.html) for platform-specific dependency installation and [Quick Start](https://trx-cpp.readthedocs.io/en/latest/quick_start.html) for a complete first program.

## Documentation

Full documentation is at **[trx-cpp.readthedocs.io](https://trx-cpp.readthedocs.io/en/latest/)**.

## Third-party notices

- `mio` by Martin Andreyel (https://github.com/mandreyel/mio) is vendored in
  `third_party/mio`. See `third_party/mio/LICENSE` for the license text.
- `json11` by Dropbox (https://github.com/dropbox/json11) is vendored in
  `third_party/json11`. See `third_party/json11/LICENSE` for the license text.
- `clang-format-all` and `check_syntax` are adapted from MRtrix3
  (https://github.com/MRtrix3/mrtrix3) and retain their original copyright
  notices under the Mozilla Public License 2.0.

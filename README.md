# TRX-cpp
[![codecov](https://codecov.io/gh/tee-ar-ex/trx-cpp/branch/main/graph/badge.svg)](https://codecov.io/gh/tee-ar-ex/trx-cpp)

TRX-cpp is a C++11 library for reading, writing, and memory-mapping the TRX
tractography file format (zip archives or on-disk directories of memmaps).

## Documentation

Project documentation (build/usage instructions and API reference) is hosted at
https://trx-cpp.readthedocs.io/en/latest/.


## Third-party notices

- `mio` by Martin Andreyel (https://github.com/mandreyel/mio) is vendored in
  `third_party/mio`. See `third_party/mio/LICENSE` for the license text.
- `json11` by Dropbox (https://github.com/dropbox/json11) is vendored in
  `third_party/json11`. See `third_party/json11/LICENSE` for the license text.
- `clang-format-all` and `check_syntax` are adapted from MRtrix3
  (https://github.com/MRtrix3/mrtrix3) and retain their original copyright
  notices under the Mozilla Public License 2.0.

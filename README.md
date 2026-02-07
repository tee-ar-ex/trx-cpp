# TRX-cpp
[![codecov](https://codecov.io/gh/tee-ar-ex/trx-cpp/branch/main/graph/badge.svg)](https://codecov.io/gh/tee-ar-ex/trx-cpp)

TRX-cpp is a C++11 library for reading, writing, and memory-mapping the TRX
tractography file format (zip archives or on-disk directories of memmaps).

## Documentation

Project documentation lives in `docs/` and includes build/usage instructions
plus the API reference. Build the site locally with the `docs` CMake target:

```
# Install prerequisites (Ubuntu example)
sudo apt-get install -y doxygen python3-pip
python3 -m pip install --user -r docs/requirements.txt

# Configure once, then build the docs target
cmake -S . -B build
cmake --build build --target docs
```

Open `docs/_build/html/index.html` in a browser.


## Third-party notices

- `mio` by Martin Andreyel (https://github.com/mandreyel/mio) is vendored in
  `third_party/mio`. See `third_party/mio/LICENSE` for the license text.
- `json11` by Dropbox (https://github.com/dropbox/json11) is vendored in
  `third_party/json11`. See `third_party/json11/LICENSE` for the license text.
- `clang-format-all` and `check_syntax` are adapted from MRtrix3
  (https://github.com/MRtrix3/mrtrix3) and retain their original copyright
  notices under the Mozilla Public License 2.0.

## Usage Examples

See the documentation in `docs/` for examples and API details.


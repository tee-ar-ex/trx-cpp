# TRX-cpp

The C++ implementations to the memory-mapped tractography file format.

## Installation
### Dependencies

The project dependencies are handled using [Conan](https://conan.io/).
Its installation can be found in their [documentation](https://docs.conan.io/en/latest/installation.html).
It is basically `pip install conan`, for Python >= 3.6.

To install the dependencies, run the following:
```bash
mkdir build && cd build
conan install ..
```

### Building

In `./build`:
```bash
cmake .. && make
```

### Running tests

In `./build`:
```bash
bin/tests
```

## How to use
COMING SOON

Examples to set up:
- Adding to my project without cmake?
- Adding to my project without conan
- Adding to my project with conan

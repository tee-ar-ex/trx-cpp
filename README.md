# TRX-cpp

The C++ implementations to the memory-mapped tractography file format.

## Installation

The project requires `gcc`, `make` and `cmake`. Please refer to your OS instructions
for proper installation.

### Dependencies

The project dependencies are handled using [Conan](https://conan.io/).
Its installation can be found in their [documentation](https://docs.conan.io/en/latest/installation.html).
TL;DR: It is basically `pip install conan`, and works for Python >= 3.6.

To install the dependencies, run the following:
```bash
# Fresh build
rm -Rf build && mkdir build && cd build

# Creates a Conan profile relative to the project
conan profile new ./.conan --detect && conan profile update settings.compiler.libcxx=libstdc++11 ./.conan

# Install dependencies
conan install --build=missing --settings=build_type=Debug --profile ./.conan ..

# Build
cmake -S .. -B . && make
```
### Running tests

In `./build`:
```bash
bin/tests
```

### Iterating the code with new changes

In `./build`:
```bash
make && bin/tests
```

## How to use
COMING SOON

Examples to set up:
- Adding to my project without cmake?
- Adding to my project without conan
- Adding to my project with conan

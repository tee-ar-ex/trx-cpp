Building
========

Dependencies
------------

Required:

- C++17 compiler
- libzip
- Eigen3


Installing dependencies:
------------------------

These examples include installing google test,
but this is only necessary if you want to build the tests.
Similarly, ninja is not strictly necessary, but it is recommended.

On Debian-based systems the zip tools have been split into separate packages
on recent ubuntu versions.

.. code-block:: bash

   sudo apt-get install \
      zlib1g-dev \
      libeigen3-dev \
      libzip-dev \
      zipcmp \
      zipmerge \
      ziptool \
      ninja-build \
      libgtest-dev

On Mac OS, you can install the dependencies with brew:

.. code-block:: bash

   brew install libzip eigen googletest ninja


On Windows, you can install the dependencies through vcpkg and chocolatey:

.. code-block:: powershell
   choco install ninja -y
   vcpkg install libzip eigen3 gtest


Building to use in other projects
---------------------------------

.. code-block:: bash

   cmake -S . -B build \
     -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DTRX_BUILD_EXAMPLES=OFF \
     -DTRX_ENABLE_INSTALL=ON \
     -DCMAKE_INSTALL_PREFIX=/path/to/installation/directory
   cmake --build build --config Release
   cmake --install build

Key CMake options:

- ``TRX_ENABLE_INSTALL``: Install package config and targets (default ON for top-level builds)
- ``TRX_BUILD_EXAMPLES``: Build example executables (default ON)
- ``TRX_BUILD_TESTS``: Build tests (default OFF)
- ``TRX_BUILD_DOCS``: Build docs with Doxygen/Sphinx (default OFF)
- ``TRX_ENABLE_CLANG_TIDY``: Run clang-tidy during builds (default OFF)
- ``TRX_USE_CONAN```: Use Conan setup in ```cmake/ConanSetup.cmake`` (default OFF)

To use trx-cpp from another CMake project after installation:

.. code-block:: cmake

   find_package(trx-cpp CONFIG REQUIRED)
   target_link_libraries(your_target PRIVATE trx-cpp::trx)

If you prefer vendoring trx-cpp, you can add it as a subdirectory and link the
target directly:

.. code-block:: cmake

   add_subdirectory(path/to/trx-cpp)
   target_link_libraries(your_target PRIVATE trx-cpp::trx)


Building for testing
--------------------

.. code-block:: bash

   cmake -S . -B build \
     -G Ninja \
     -DTRX_BUILD_TESTS=ON \
     -DTRX_BUILD_EXAMPLES=OFF

   cmake --build build
   ctest --test-dir build --output-on-failure

Tests require GTest to be discoverable by CMake (e.g., via a system package or
``GTest_DIR``). If GTest is not found, tests will be skipped.


Building documentation:
-----------------------

Building the docs requires both Doxygen and ``sphinx-build`` on your PATH.

.. code-block:: bash

   cmake -S . -B build \
     -G Ninja \
     -DTRX_BUILD_DOCS=ON

   cmake --build build --target docs

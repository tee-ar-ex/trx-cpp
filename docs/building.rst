Building
========

Dependencies
------------

Required:

- C++17 compiler
- zlib (``zlib1g-dev`` / ``zlib-devel`` / Homebrew ``zlib``)

libzip and Eigen 3.4+ are fetched automatically by CMake when not found
locally — no manual installation required.


Installing dependencies
------------------------

The examples below include GoogleTest, which is only required when building
the tests. Ninja is optional but recommended.

On Debian-based systems:

.. code-block:: bash

   sudo apt-get install \
      zlib1g-dev \
      ninja-build \
      libgtest-dev

On Mac OS, you can install the dependencies with brew:

.. code-block:: bash

   brew install googletest ninja zlib


On Windows, you can install the dependencies through vcpkg and chocolatey:

.. code-block:: powershell

   choco install ninja -y
   vcpkg install gtest zlib


Building to use in other projects
---------------------------------

The recommended way to consume trx-cpp is via ``add_subdirectory`` or
FetchContent — this always works regardless of how libzip is resolved.
When using either approach, **zlib is the only dependency you need
pre-installed**; libzip and Eigen 3.4+ are fetched automatically if not
already present.  Because Eigen is a public dependency of trx-cpp, your
code can use ``Eigen3::Eigen`` (and include Eigen headers) without a
separate ``find_package(Eigen3)`` call.

.. code-block:: cmake

   # CMakeLists.txt of your project
   add_subdirectory(path/to/trx-cpp)           # vendored copy
   target_link_libraries(your_target PRIVATE trx-cpp::trx)

   # — or via FetchContent —
   include(FetchContent)
   FetchContent_Declare(trx-cpp
       GIT_REPOSITORY https://github.com/tee-ar-ex/trx-cpp.git
       GIT_TAG        main)
   FetchContent_MakeAvailable(trx-cpp)
   target_link_libraries(your_target PRIVATE trx-cpp::trx)

**Installing trx-cpp** (``cmake --install``) requires libzip to be available
as a system-installed package with CMake config support.  When libzip is
auto-fetched by CMake (the default), the install step is automatically skipped
with a status message.

To enable installation, first install libzip through your package manager:

.. code-block:: bash

   # Debian/Ubuntu
   sudo apt-get install libzip-dev

   # macOS
   brew install libzip

.. code-block:: powershell

   # Windows (vcpkg)
   vcpkg install libzip

Then configure and install trx-cpp normally:

.. code-block:: bash

   cmake -S . -B build \
     -G Ninja \
     -DCMAKE_BUILD_TYPE=Release \
     -DTRX_BUILD_EXAMPLES=OFF \
     -DTRX_ENABLE_INSTALL=ON \
     -DCMAKE_INSTALL_PREFIX=/path/to/installation/directory
   cmake --build build --config Release
   cmake --install build

After installation, consume the package with:

.. code-block:: cmake

   find_package(trx-cpp CONFIG REQUIRED)
   target_link_libraries(your_target PRIVATE trx-cpp::trx)

Key CMake options:

- ``TRX_ENABLE_INSTALL``: Install package config and targets (default ON for top-level builds)
- ``TRX_BUILD_EXAMPLES``: Build example executables (default ON)
- ``TRX_BUILD_TESTS``: Build tests (default OFF)
- ``TRX_BUILD_DOCS``: Build docs with Doxygen/Sphinx (default OFF)
- ``TRX_ENABLE_CLANG_TIDY``: Run clang-tidy during builds (default OFF)
- ``TRX_USE_CONAN``: Use Conan setup in ``cmake/ConanSetup.cmake`` (default OFF)
- ``TRX_FETCH_EIGEN``: Fetch Eigen3 with FetchContent when not found locally (default ON)


Building for testing
--------------------

.. code-block:: bash

   cmake -S . -B build \
     -G Ninja \
     -DTRX_BUILD_TESTS=ON \
     -DTRX_ENABLE_NIFTI=ON \
     -DTRX_BUILD_EXAMPLES=OFF

   cmake --build build
   ctest --test-dir build --output-on-failure

Tests require GTest to be discoverable by CMake (e.g., via a system package or
``GTest_DIR``). If GTest is not found, tests will be skipped.
zlib must be discoverable by CMake (``ZLIB::ZLIB``), including for NIfTI I/O.


Building documentation:
-----------------------

Building the docs requires both Doxygen and ``sphinx-build`` on your PATH.

.. code-block:: bash

   cmake -S . -B build \
     -G Ninja \
     -DTRX_BUILD_DOCS=ON

   cmake --build build --target docs

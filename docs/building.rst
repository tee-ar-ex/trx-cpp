Building
========

Dependencies
------------

Required:

- C++17 compiler and CMake (>= 3.20)
- libzip
- Eigen3

Optional:

- GTest (for building tests)

Quick build (no tests)
----------------------

.. code-block:: bash

   cmake -S . -B build
   cmake --build build

Build with tests
----------------

.. code-block:: bash

   cmake -S . -B build \
     -DTRX_BUILD_TESTS=ON \
     -DGTest_DIR=/path/to/GTestConfig.cmake
   cmake --build build
   ctest --test-dir build --output-on-failure

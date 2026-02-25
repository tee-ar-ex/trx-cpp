Quick Start
===========

This page walks through getting trx-cpp installed and writing a minimal
program that loads a TRX file, prints basic statistics, and saves a copy.

Prerequisites
-------------

- A C++17 compiler (GCC ≥ 7, Clang ≥ 5, MSVC 2019+)
- CMake ≥ 3.14
- `libzip <https://libzip.org/>`_
- `Eigen 3.4+ <https://eigen.tuxfamily.org/>`_

See :doc:`building` for platform-specific installation commands.

Install trx-cpp
---------------

Build and install the library so it can be found by ``find_package``:

.. code-block:: bash

   git clone https://github.com/tee-ar-ex/trx-cpp.git
   cmake -S trx-cpp -B trx-cpp/build \
     -DCMAKE_BUILD_TYPE=Release \
     -DTRX_BUILD_EXAMPLES=OFF \
     -DTRX_ENABLE_INSTALL=ON \
     -DCMAKE_INSTALL_PREFIX=$HOME/.local
   cmake --build trx-cpp/build --config Release
   cmake --install trx-cpp/build

Alternatively, add trx-cpp as a subdirectory in your project (no install step
needed):

.. code-block:: cmake

   add_subdirectory(path/to/trx-cpp)
   target_link_libraries(my_app PRIVATE trx-cpp::trx)

Write a first program
---------------------

Create a ``CMakeLists.txt`` and a ``main.cpp``:

.. code-block:: cmake

   # CMakeLists.txt
   cmake_minimum_required(VERSION 3.14)
   project(hello_trx)

   find_package(trx-cpp CONFIG REQUIRED)

   add_executable(hello_trx main.cpp)
   target_link_libraries(hello_trx PRIVATE trx-cpp::trx)

.. code-block:: cpp

   // main.cpp
   #include <trx/trx.h>
   #include <iostream>

   int main(int argc, char* argv[]) {
     if (argc < 2) {
       std::cerr << "usage: hello_trx <file.trx>\n";
       return 1;
     }

     auto trx = trx::load_any(argv[1]);

     std::cout << "streamlines : " << trx.num_streamlines() << "\n";
     std::cout << "vertices    : " << trx.num_vertices()    << "\n";
     std::cout << "dtype       : " << trx.positions.dtype   << "\n";

     for (const auto& [name, arr] : trx.data_per_streamline) {
       std::cout << "dps/" << name
                 << "  (" << arr.rows() << " x " << arr.cols() << ")\n";
     }

     trx.close();
     return 0;
   }

Build and run:

.. code-block:: bash

   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
   cmake --build build
   ./build/hello_trx /path/to/tracks.trx

Next steps
----------

- :doc:`reading` — access streamline positions and metadata
- :doc:`writing` — create and save TRX files
- :doc:`streaming` — stream streamlines without buffering the full dataset
- :doc:`api_layers` — understand the three API layers

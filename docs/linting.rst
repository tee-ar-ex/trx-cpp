Linting and Style Checks
========================

This repo includes `.clang-tidy` and `.clang-format` at the root, plus
MRtrix-inspired helper scripts for formatting and style checks.

Prerequisites
-------------

- `clang-format` available on `PATH`
  - macOS (Homebrew): `brew install llvm` (or `llvm@17`) and ensure
    `clang-format` is on `PATH`
  - Ubuntu: `sudo apt-get install clang-format`
- For `check_syntax` on macOS, GNU grep is required (`brew install grep`,
  then it will use `ggrep`).

clang-format (bulk formatting)
------------------------------

.. code-block:: bash

   ./clang-format-all

You can target a specific clang-format binary:

.. code-block:: bash

   ./clang-format-all --executable /path/to/clang-format

check_syntax (style rules)
--------------------------

Run the MRtrix-style checks against the C++ sources:

.. code-block:: bash

   ./check_syntax

Results are written to `syntax.log` when issues are found.

clang-tidy
----------

Generate a build with compile commands, then run clang-tidy (matches CI):

.. code-block:: bash

   cmake -S . -B build \
     -DTRX_USE_CONAN=OFF \
     -DTRX_BUILD_EXAMPLES=ON \
     -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
   run-clang-tidy -p build $(git ls-files '*.cpp' '*.h' '*.hpp' '*.tpp' ':!third_party/**')

To run clang-tidy automatically during builds:

.. code-block:: bash

   cmake -S . -B build -DTRX_ENABLE_CLANG_TIDY=ON
   cmake --build build

If you have `run-clang-tidy` installed (LLVM extras), you can lint everything
tracked by the repo (excluding `third_party`), which matches CI.

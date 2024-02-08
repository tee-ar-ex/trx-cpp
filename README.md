# TRX-cpp

The C++ implementations to the memory-mapped tractography file format.

## Installation

### Dependencies

- c++11 compiler / cmake
- libzip
- nlohmann::json
- Eigen3
- spdlog
- GTest (optional)
- mio

### Installing

`cmake . && make`

## How to use

* Install all dependencies (after either cloning github repo or downloading) using these commands
  * `mkdir build`
  * `cd build`
  * `cmake ..`
  * `make`
  * `make install`
* clone into this repo using `git clone https://github.com/tee-ar-ex/trx-cpp.git`
* To run trx_cpp:
  * `cmake -B build`
  * `cd build`
  * `make`

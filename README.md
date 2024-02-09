# TRX-cpp

The C++ implementations to the memory-mapped tractography file format.

## Installation

### Dependencies

- c++11 compiler / cmake
- libzip (version 1.10.1 downloaded from https://libzip.org/download/)
- nlohmann::json(version 3.11.3 cloned from https://github.com/nlohmann/json)
- Eigen3(version 3.4.0 from https://eigen.tuxfamily.org/index.php?title=Main_Page )
- spdlog(spd) [version: 1.13.0 cloned from https://github.com/gabime/spdlog)
- GTest (optional)
- mio (cloned from fhttps://github.com/vimpunk/mio/tree/master)

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

cmake_minimum_required(VERSION 3.0.0)
project(trx_example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

add_executable(load_trx load_trx.cpp)

set(TRX_USE_CONAN OFF)
add_subdirectory(vendor/trx)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

cmake_minimum_required(VERSION 3.16)
project(trx_cpp_test_package LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(trx-cpp CONFIG REQUIRED)

add_executable(example src/example.cpp)
target_link_libraries(example PRIVATE trx-cpp::trx)

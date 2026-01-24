cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)
project(trx VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 17)

#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Debug)

find_package(libzip REQUIRED)
find_package (Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_path(MIO_INCLUDE_DIR mio/mmap.hpp PATH_SUFFIXES include)
if (NOT MIO_INCLUDE_DIR)
	message(FATAL_ERROR "mio headers not found. Set MIO_INCLUDE_DIR to the folder containing mio/mmap.hpp.")
endif()

add_library(trx src/trx.cpp src/trx.tpp src/trx.h)

TARGET_LINK_LIBRARIES(trx
	PRIVATE
	nlohmann_json::nlohmann_json
	libzip::zip
	Eigen3::Eigen
    spdlog::spdlog
    spdlog::spdlog_header_only
)
target_include_directories(trx PRIVATE ${MIO_INCLUDE_DIR})


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

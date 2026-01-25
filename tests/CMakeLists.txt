cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)
project(trx)
set (CMAKE_CXX_STANDARD 17)

set(PROJECT_BINARY_DIR ../../builds)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/tests)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/build/tests)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

set(CMAKE_BUILD_TYPE Debug)

find_package(libzip REQUIRED)
find_package (Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
option(TRX_ENABLE_TESTS "Build trx-cpp tests" ON)
find_package(GTest CONFIG QUIET)
if (TRX_ENABLE_TESTS AND NOT GTest_FOUND)
	message(FATAL_ERROR "GTest not found. Set GTest_DIR/CMAKE_PREFIX_PATH or pass -DTRX_ENABLE_TESTS=OFF to skip tests.")
endif()
find_path(MIO_INCLUDE_DIR mio/mmap.hpp PATH_SUFFIXES include)
if (NOT MIO_INCLUDE_DIR)
	message(FATAL_ERROR "mio headers not found. Set MIO_INCLUDE_DIR to the folder containing mio/mmap.hpp.")
endif()

if (TRX_ENABLE_TESTS AND GTest_FOUND)
	enable_testing()

	include_directories(../src)
	add_executable(test_mmap test_trx_mmap.cpp ../src/trx.h ../src/trx.tpp ../src/trx.cpp)


	TARGET_LINK_LIBRARIES(test_mmap
		nlohmann_json::nlohmann_json
		Eigen3::Eigen
		libzip::zip
		GTest::gtest
		GTest::gtest_main
		spdlog::spdlog
		spdlog::spdlog_header_only
	)
	target_include_directories(test_mmap PRIVATE ${MIO_INCLUDE_DIR})


	set(CPACK_PROJECT_NAME ${PROJECT_NAME})
	set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
	include(CPack)
	include(GoogleTest)
	gtest_discover_tests(test_mmap)
endif()

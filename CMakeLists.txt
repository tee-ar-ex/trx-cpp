cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)
project(trx VERSION 0.1.0)
set (CMAKE_CXX_STANDARD 17)

option(TRX_USE_CONAN "Should Conan package manager be used?" ON)
option(TRX_BUILD_TESTS "Build trx tests" ON)

#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Debug)

find_package(libzip REQUIRED)
find_package(Eigen3 CONFIG QUIET)
if (NOT Eigen3_FOUND)
    find_package(Eigen3 REQUIRED) # try module mode
endif()
# Create an imported target if the package did not provide one (module mode)
if (NOT TARGET Eigen3::Eigen AND EXISTS "${EIGEN3_INCLUDE_DIR}")
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
    )
endif()
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_path(MIO_INCLUDE_DIR mio/mmap.hpp PATH_SUFFIXES include)
if (NOT MIO_INCLUDE_DIR)
	message(FATAL_ERROR "mio headers not found. Set MIO_INCLUDE_DIR to the folder containing mio/mmap.hpp.")
endif()

add_library(trx src/trx.cpp src/trx.tpp src/trx.h)

if(TRX_USE_CONAN AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConanSetup.cmake")
    include(cmake/ConanSetup.cmake)
elseif(TRX_USE_CONAN)
    message(STATUS "TRX_USE_CONAN enabled but cmake/ConanSetup.cmake not found; skipping Conan.")
endif()

TARGET_LINK_LIBRARIES(trx
	PUBLIC
		nlohmann_json::nlohmann_json
		libzip::zip
		Eigen3::Eigen
        spdlog::spdlog
        spdlog::spdlog_header_only
)
target_include_directories(trx
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${MIO_INCLUDE_DIR}
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

if(TRX_BUILD_TESTS)
    find_package(GTest CONFIG QUIET)
    if(NOT GTest_FOUND)
        find_package(GTest QUIET)
    endif()
    if(GTest_FOUND)
        enable_testing()
        add_subdirectory(tests)
    else()
        message(STATUS "GTest not found; skipping tests. Set GTest_DIR to a config path to enable.")
    endif()
endif()


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

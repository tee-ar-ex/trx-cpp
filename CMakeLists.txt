cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)

project(trx VERSION 0.1.0)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(TRX_USE_CONAN "Should Conan package manager be used?" OFF)
option(TRX_BUILD_TESTS "Build trx tests" OFF)
option(TRX_BUILD_EXAMPLES "Build trx example commandline programs" ON)

if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
endif()

find_package(libzip REQUIRED)
set(TRX_LIBZIP_TARGET "")
if(TARGET libzip::libzip)
    set(TRX_LIBZIP_TARGET libzip::libzip)
elseif(TARGET libzip::zip)
    set(TRX_LIBZIP_TARGET libzip::zip)
elseif(TARGET zip::zip)
    set(TRX_LIBZIP_TARGET zip::zip)
else()
    message(FATAL_ERROR "No suitable libzip target (expected libzip::libzip or zip::zip)")
endif()
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
set(TRX_HAVE_MIO_TARGET OFF)
set(MIO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mio/include")
if (NOT EXISTS "${MIO_INCLUDE_DIR}/mio/mmap.hpp")
    message(FATAL_ERROR "Vendored mio headers not found at ${MIO_INCLUDE_DIR}/mio/mmap.hpp.")
endif()

add_library(trx
    src/trx.cpp
    include/trx/trx.h
    include/trx/trx.tpp
    third_party/json11/json11.cpp
)
add_library(trx-cpp::trx ALIAS trx)

target_compile_features(trx PUBLIC cxx_std_17)

if(TRX_USE_CONAN AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConanSetup.cmake")
    include(cmake/ConanSetup.cmake)
elseif(TRX_USE_CONAN)
    message(STATUS "TRX_USE_CONAN enabled but cmake/ConanSetup.cmake not found; skipping Conan.")
endif()

# Fallback for libzip packages that don't expose include dirs via CMake targets.
set(TRX_LIBZIP_INCLUDE_DIR "")
get_target_property(_trx_libzip_includes ${TRX_LIBZIP_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
if(NOT _trx_libzip_includes)
    find_path(TRX_LIBZIP_INCLUDE_DIR zip.h)
    if(NOT TRX_LIBZIP_INCLUDE_DIR)
        message(FATAL_ERROR "libzip headers not found. Set TRX_LIBZIP_INCLUDE_DIR or fix libzip CMake targets.")
    endif()
endif()

TARGET_LINK_LIBRARIES(trx
    PUBLIC
        ${TRX_LIBZIP_TARGET}
        Eigen3::Eigen
        $<$<BOOL:${TRX_HAVE_MIO_TARGET}>:mio::mio>
)
target_include_directories(trx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<$<NOT:$<BOOL:${TRX_HAVE_MIO_TARGET}>>:$<BUILD_INTERFACE:${MIO_INCLUDE_DIR}>>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/json11>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
if(TRX_LIBZIP_INCLUDE_DIR)
    target_include_directories(trx PUBLIC
        $<BUILD_INTERFACE:${TRX_LIBZIP_INCLUDE_DIR}>
        $<INSTALL_INTERFACE:${TRX_LIBZIP_INCLUDE_DIR}>
    )
endif()

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

if(TRX_BUILD_EXAMPLES)
    FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v3.2.0
    )
    FetchContent_MakeAvailable(cxxopts)
    add_subdirectory(examples)
endif()

# Installation and package config
set(TRX_INSTALL_CONFIGDIR "${CMAKE_INSTALL_LIBDIR}/cmake/trx-cpp")

install(TARGETS trx
    EXPORT trx-cppTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY third_party/mio/include/mio DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES third_party/json11/json11.hpp DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

install(EXPORT trx-cppTargets
    FILE trx-cppTargets.cmake
    NAMESPACE trx-cpp::
    DESTINATION ${TRX_INSTALL_CONFIGDIR}
)

configure_package_config_file(
    cmake/trx-cppConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/trx-cppConfig.cmake
    INSTALL_DESTINATION ${TRX_INSTALL_CONFIGDIR}
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/trx-cppConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMinorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/trx-cppConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/trx-cppConfigVersion.cmake
    DESTINATION ${TRX_INSTALL_CONFIGDIR}
)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

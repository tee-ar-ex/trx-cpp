cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)

project(trx VERSION 0.1.0)

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
include(FetchContent)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

set(TRX_IS_TOP_LEVEL OFF)
if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(TRX_IS_TOP_LEVEL ON)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(TRX_USE_CONAN "Should Conan package manager be used?" OFF)
option(TRX_BUILD_TESTS "Build trx tests" OFF)
option(TRX_BUILD_EXAMPLES "Build trx example commandline programs" ON)
option(TRX_BUILD_BENCHMARKS "Build trx benchmarks" OFF)
option(TRX_ENABLE_CLANG_TIDY "Run clang-tidy during builds" OFF)
option(TRX_ENABLE_INSTALL "Install trx-cpp targets" ${TRX_IS_TOP_LEVEL})
option(TRX_BUILD_DOCS "Build API documentation with Doxygen/Sphinx" OFF)
option(TRX_ENABLE_NIFTI "Enable optional NIfTI header utilities" ${TRX_BUILD_EXAMPLES})
option(TRX_FETCH_EIGEN "Fetch Eigen3 via FetchContent if not found" ON)

if(TRX_ENABLE_CLANG_TIDY)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
    find_program(TRX_CLANG_TIDY_EXE NAMES clang-tidy)
    if(TRX_CLANG_TIDY_EXE)
        set(CMAKE_CXX_CLANG_TIDY "${TRX_CLANG_TIDY_EXE}")
        if(APPLE)
            execute_process(
                COMMAND xcrun --show-sdk-path
                OUTPUT_VARIABLE TRX_APPLE_SDK_PATH
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            if(TRX_APPLE_SDK_PATH)
                if(NOT CMAKE_OSX_SYSROOT)
                    set(CMAKE_OSX_SYSROOT "${TRX_APPLE_SDK_PATH}" CACHE STRING "macOS SDK" FORCE)
                endif()
                list(APPEND CMAKE_CXX_CLANG_TIDY "--extra-arg=-isysroot${TRX_APPLE_SDK_PATH}")
            endif()
        endif()
    else()
        message(WARNING "TRX_ENABLE_CLANG_TIDY is ON but clang-tidy was not found.")
    endif()
endif()

if(MSVC)
    add_compile_definitions(_CRT_SECURE_NO_WARNINGS)
    add_compile_definitions(NOMINMAX)
endif()

if(TRX_USE_CONAN AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cmake/ConanSetup.cmake")
    include(cmake/ConanSetup.cmake)
elseif(TRX_USE_CONAN)
    message(STATUS "TRX_USE_CONAN enabled but cmake/ConanSetup.cmake not found; skipping Conan.")
endif()

# ── libzip ──────────────────────────────────────────────────────────────────
# TRX_LIBZIP_TARGET: a caller (e.g. ITKTractographyTRX) may pre-set this to
# a specific target to skip libzip discovery entirely. Plain variable (not
# CACHE) so that a parent scope's set() propagates into this FetchContent
# child scope without being cleared by CMake's CACHE precedence rules.
if(TRX_LIBZIP_TARGET AND TARGET ${TRX_LIBZIP_TARGET})
    message(STATUS "trx-cpp: using provided libzip target: ${TRX_LIBZIP_TARGET}")
else()
    # CONFIG-only: only accept a system libzip that provides a proper cmake
    # config (creating libzip::zip). If not found, always fetch and build.
    find_package(libzip CONFIG QUIET)
    if(NOT TARGET libzip::zip)
        message(STATUS "libzip not found via config-mode find_package; fetching v1.11.4")
        set(LIBZIP_DO_INSTALL OFF)
        set(BUILD_TOOLS OFF)
        set(BUILD_REGRESS OFF)
        set(BUILD_EXAMPLES OFF)
        set(BUILD_DOC OFF)
        # TRX files only use deflate; disable optional codecs to avoid pulling
        # in system bzip2/lzma/zstd as undeclared link dependencies.
        # CMAKE_POLICY_DEFAULT_CMP0077 propagates into the subdirectory scope
        # that FetchContent_MakeAvailable creates, so libzip's option() calls
        # respect these normal variables.
        set(CMAKE_POLICY_DEFAULT_CMP0077 NEW)
        set(ENABLE_BZIP2 OFF)
        set(ENABLE_LZMA OFF)
        set(ENABLE_ZSTD OFF)
        set(_saved_BUILD_SHARED_LIBS ${BUILD_SHARED_LIBS})
        set(BUILD_SHARED_LIBS OFF)
        FetchContent_Declare(
            libzip
            GIT_REPOSITORY https://github.com/nih-at/libzip.git
            GIT_TAG v1.11.4
        )
        FetchContent_MakeAvailable(libzip)
        set(BUILD_SHARED_LIBS ${_saved_BUILD_SHARED_LIBS})
        unset(_saved_BUILD_SHARED_LIBS)
        if(TARGET zip)
            set_target_properties(zip PROPERTIES POSITION_INDEPENDENT_CODE ON)
        endif()
    endif()
    # Detect the target name: CONFIG find_package creates libzip::zip;
    # FetchContent creates the bare 'zip' target.
    if(TARGET libzip::zip)
        set(TRX_LIBZIP_TARGET libzip::zip)
    elseif(TARGET zip)
        set(TRX_LIBZIP_TARGET zip)
    else()
        message(FATAL_ERROR "No suitable libzip target found (expected libzip::zip or zip)")
    endif()
endif()

# ── Eigen ───────────────────────────────────────────────────────────────────
# TRX_EIGEN3_TARGET: a caller may pre-set this to a specific Eigen target
# (e.g. ITK::ITKEigen3Module) to skip find_package(Eigen3). Plain variable
# (not CACHE) for the same reason as TRX_LIBZIP_TARGET above.
if(TRX_EIGEN3_TARGET AND TARGET ${TRX_EIGEN3_TARGET})
    message(STATUS "trx-cpp: using provided Eigen3 target: ${TRX_EIGEN3_TARGET}")
else()
    if(Eigen3_ROOT)
        list(PREPEND CMAKE_PREFIX_PATH "${Eigen3_ROOT}")
    endif()
    find_package(Eigen3 3.4 CONFIG QUIET)
    if(NOT TARGET Eigen3::Eigen)
        # Try module mode for distributions that ship FindEigen3.cmake.
        find_package(Eigen3 3.4 QUIET MODULE)
    endif()
    if(NOT TARGET Eigen3::Eigen AND TRX_FETCH_EIGEN)
        message(STATUS "Eigen3 not found; fetching v3.4.0")
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "Disable Eigen docs" FORCE)
        set(EIGEN_BUILD_TESTING OFF CACHE BOOL "Disable Eigen tests" FORCE)
        set(EIGEN_BUILD_PKGCONFIG OFF CACHE BOOL "Disable Eigen pkg-config generation" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "Disable Eigen testing" FORCE)
        FetchContent_Declare(
            eigen
            URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
            DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        )
        FetchContent_MakeAvailable(eigen)
    endif()
    # Create an imported target if the package did not provide one (module mode)
    if(NOT TARGET Eigen3::Eigen AND EXISTS "${EIGEN3_INCLUDE_DIR}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    endif()
    if(NOT TARGET Eigen3::Eigen)
        message(FATAL_ERROR "Eigen3 target not found. Install Eigen3 or set TRX_FETCH_EIGEN=ON.")
    endif()
    set(TRX_EIGEN3_TARGET Eigen3::Eigen)
endif()

# ── Vendored headers ────────────────────────────────────────────────────────
set(MIO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mio/include")
if(NOT EXISTS "${MIO_INCLUDE_DIR}/mio/mmap.hpp")
    message(FATAL_ERROR "Vendored mio headers not found at ${MIO_INCLUDE_DIR}/mio/mmap.hpp.")
endif()

# ── Core library ────────────────────────────────────────────────────────────
add_library(trx
    src/trx.cpp
    src/detail/dtype_helpers.cpp
    include/trx/trx.h
    include/trx/trx.tpp
    third_party/json11/json11.cpp
)
set_source_files_properties(third_party/json11/json11.cpp PROPERTIES SKIP_LINTING ON)
add_library(trx-cpp::trx ALIAS trx)

target_compile_features(trx PUBLIC cxx_std_17)

target_link_libraries(trx
    PUBLIC
        ${TRX_LIBZIP_TARGET}
        ${TRX_EIGEN3_TARGET}
)
target_include_directories(trx
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        $<BUILD_INTERFACE:${MIO_INCLUDE_DIR}>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/third_party/json11>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)
# Fallback for system libzip packages that don't expose include dirs via
# their CMake targets. Only needed for distro packages with broken configs.
get_target_property(_trx_libzip_includes ${TRX_LIBZIP_TARGET} INTERFACE_INCLUDE_DIRECTORIES)
if(NOT _trx_libzip_includes)
    find_path(TRX_LIBZIP_INCLUDE_DIR zip.h)
    if(TRX_LIBZIP_INCLUDE_DIR)
        target_include_directories(trx PUBLIC
            $<BUILD_INTERFACE:${TRX_LIBZIP_INCLUDE_DIR}>
        )
    else()
        message(FATAL_ERROR "libzip headers not found. Set TRX_LIBZIP_INCLUDE_DIR or fix libzip CMake targets.")
    endif()
endif()
unset(_trx_libzip_includes)

# ── Tests ───────────────────────────────────────────────────────────────────
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

# ── Benchmarks ──────────────────────────────────────────────────────────────
if(TRX_BUILD_BENCHMARKS)
    find_package(benchmark CONFIG QUIET)
    if(NOT benchmark_FOUND)
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable benchmark gtest" FORCE)
        FetchContent_Declare(
            benchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG v1.8.3
        )
        FetchContent_MakeAvailable(benchmark)
    endif()
    add_subdirectory(bench)
endif()

# ── Optional NIfTI utilities ────────────────────────────────────────────────
if(TRX_ENABLE_NIFTI)
    # TRX_ZLIB_TARGET: a caller may pre-set this to skip find_package(ZLIB)
    # (e.g. ITK::ITKZLIBModule). Plain variable, same convention as above.
    if(TRX_ZLIB_TARGET AND TARGET ${TRX_ZLIB_TARGET})
        message(STATUS "trx-cpp: using provided ZLIB target: ${TRX_ZLIB_TARGET}")
    else()
        find_package(ZLIB REQUIRED)
        set(TRX_ZLIB_TARGET ZLIB::ZLIB)
    endif()
    add_library(trx-nifti
        src/nifti_io.cpp
    )
    add_library(trx-cpp::trx-nifti ALIAS trx-nifti)
    target_compile_features(trx-nifti PUBLIC cxx_std_17)
    target_include_directories(trx-nifti
        PUBLIC
            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
    )
    target_link_libraries(trx-nifti
        PUBLIC
            ${TRX_EIGEN3_TARGET}
            ${TRX_ZLIB_TARGET}
    )
endif()

# ── Examples ────────────────────────────────────────────────────────────────
if(TRX_BUILD_EXAMPLES)
    FetchContent_Declare(
        cxxopts
        GIT_REPOSITORY https://github.com/jarro2783/cxxopts.git
        GIT_TAG v3.2.0
    )
    FetchContent_MakeAvailable(cxxopts)
    add_subdirectory(examples)
endif()

# ── Docs ────────────────────────────────────────────────────────────────────
if(TRX_BUILD_DOCS)
    find_package(Doxygen QUIET)
    find_program(TRX_SPHINX_BUILD_EXE NAMES sphinx-build)
    if(DOXYGEN_FOUND AND TRX_SPHINX_BUILD_EXE)
        add_custom_target(docs
            COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_CURRENT_SOURCE_DIR}/docs/_build/doxygen
            COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile
            COMMAND ${TRX_SPHINX_BUILD_EXE} -b html
                    ${CMAKE_CURRENT_SOURCE_DIR}/docs
                    ${CMAKE_CURRENT_SOURCE_DIR}/docs/_build/html
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/docs
            COMMENT "Building Doxygen XML and Sphinx HTML docs"
            VERBATIM
        )
    else()
        message(STATUS "Docs disabled: Doxygen or sphinx-build not found.")
    endif()
endif()

# ── Installation and package config ─────────────────────────────────────────
if(TRX_ENABLE_INSTALL)
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
endif()

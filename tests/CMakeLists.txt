enable_testing()

find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    find_package(GTest REQUIRED)
endif()

set(TRX_TEST_DATA_REPO "https://github.com/tee-ar-ex/trx-test-data" CACHE STRING "Git repository URL for test data.")
set(TRX_TEST_DATA_TAG "v0.1.0" CACHE STRING "Release tag for test data archives.")
set(_trx_default_archives "memmap_test_data.zip;trx_from_scratch.zip;gold_standard.zip")
set(TRX_TEST_DATA_ARCHIVES "${_trx_default_archives}" CACHE STRING "Semicolon-separated release archive names.")
# Ensure new archives are picked up even when cache already exists.
if(NOT TRX_TEST_DATA_ARCHIVES STREQUAL _trx_default_archives)
    set(TRX_TEST_DATA_ARCHIVES "${_trx_default_archives}" CACHE STRING "Semicolon-separated release archive names." FORCE)
endif()
set(TRX_TEST_DATA_DIR "${CMAKE_BINARY_DIR}/_trx_test_data" CACHE PATH "Local test data directory.")
set(TRX_TEST_DATA_ARCHIVE_DIR "${CMAKE_BINARY_DIR}/_trx_test_data_archives" CACHE PATH "Local test data archive directory.")

# Always download from releases
message(STATUS "Downloading test data from ${TRX_TEST_DATA_REPO} release ${TRX_TEST_DATA_TAG}")
file(MAKE_DIRECTORY "${TRX_TEST_DATA_DIR}")
file(MAKE_DIRECTORY "${TRX_TEST_DATA_ARCHIVE_DIR}")
foreach(_trx_archive IN LISTS TRX_TEST_DATA_ARCHIVES)
    set(_trx_archive_url "${TRX_TEST_DATA_REPO}/releases/download/${TRX_TEST_DATA_TAG}/${_trx_archive}")
    set(_trx_archive_path "${TRX_TEST_DATA_ARCHIVE_DIR}/${_trx_archive}")
    message(STATUS "Downloading ${_trx_archive}...")
    file(DOWNLOAD "${_trx_archive_url}" "${_trx_archive_path}"
        SHOW_PROGRESS
        TLS_VERIFY ON
        STATUS _download_status
    )
    list(GET _download_status 0 _status_code)
    if(NOT _status_code EQUAL 0)
        list(GET _download_status 1 _status_message)
        message(FATAL_ERROR "Failed to download ${_trx_archive}: ${_status_message}")
    endif()
    message(STATUS "Extracting ${_trx_archive}...")
    file(ARCHIVE_EXTRACT INPUT "${_trx_archive_path}" DESTINATION "${TRX_TEST_DATA_DIR}")
endforeach()
message(STATUS "Test data downloaded to ${TRX_TEST_DATA_DIR}")

add_executable(test_mmap test_trx_mmap.cpp)
target_link_libraries(test_mmap PRIVATE trx GTest::gtest_main)
target_compile_features(test_mmap PRIVATE cxx_std_14)

add_executable(test_io test_trx_io.cpp)
target_link_libraries(test_io PRIVATE trx GTest::gtest_main)
target_compile_features(test_io PRIVATE cxx_std_14)

include(GoogleTest)
gtest_discover_tests(test_mmap PROPERTIES
    ENVIRONMENT "TRX_TEST_DATA_DIR=${TRX_TEST_DATA_DIR}"
)

gtest_discover_tests(test_io PROPERTIES
    ENVIRONMENT "TRX_TEST_DATA_DIR=${TRX_TEST_DATA_DIR}"
)

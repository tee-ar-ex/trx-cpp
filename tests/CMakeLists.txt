enable_testing()

find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    find_package(GTest REQUIRED)
endif()

set(TRX_TEST_DATA_REPO "https://github.com/tee-ar-ex/trx-test-data" CACHE STRING "Git repository URL for test data.")
set(TRX_TEST_DATA_TAG "v0.1.0" CACHE STRING "Release tag for test data archives.")
option(TRX_DOWNLOAD_TEST_DATA "Download test data archives at configure time." ON)
set(_trx_default_archives "memmap_test_data.zip;trx_from_scratch.zip;gold_standard.zip")
set(TRX_TEST_DATA_ARCHIVES "${_trx_default_archives}" CACHE STRING "Semicolon-separated release archive names.")
set(TRX_TEST_DATA_DIR "${CMAKE_BINARY_DIR}/_trx_test_data" CACHE PATH "Local test data directory.")
set(TRX_TEST_DATA_ARCHIVE_DIR "${CMAKE_BINARY_DIR}/_trx_test_data_archives" CACHE PATH "Local test data archive directory.")

file(MAKE_DIRECTORY "${TRX_TEST_DATA_DIR}")
file(MAKE_DIRECTORY "${TRX_TEST_DATA_ARCHIVE_DIR}")
if(TRX_DOWNLOAD_TEST_DATA)
    message(STATUS "Configuring test data from ${TRX_TEST_DATA_REPO} release ${TRX_TEST_DATA_TAG}")
    foreach(_trx_archive IN LISTS TRX_TEST_DATA_ARCHIVES)
        set(_trx_expected_hash "")
        if(_trx_archive STREQUAL "memmap_test_data.zip")
            set(_trx_expected_hash "98ba89d7a9a7baa2d37956a0a591dce9bb4581bd01296ad5a596706ee90a52ef")
        elseif(_trx_archive STREQUAL "trx_from_scratch.zip")
            set(_trx_expected_hash "f98ab6da6a6065527fde4b0b6aa40f07583e925d952182e9bbd0febd55c0f6b2")
        elseif(_trx_archive STREQUAL "gold_standard.zip")
            set(_trx_expected_hash "35a0b633560cc2b0d8ecda885aa72d06385499e0cd1ca11a956b0904c3358f01")
        endif()
        if(NOT _trx_expected_hash)
            message(FATAL_ERROR "No SHA256 hash configured for ${_trx_archive}.")
        endif()

        set(_trx_archive_url "${TRX_TEST_DATA_REPO}/releases/download/${TRX_TEST_DATA_TAG}/${_trx_archive}")
        set(_trx_archive_path "${TRX_TEST_DATA_ARCHIVE_DIR}/${_trx_archive}")
        set(_trx_marker "${TRX_TEST_DATA_DIR}/${_trx_archive}.extracted")

        if(EXISTS "${_trx_archive_path}")
            file(SHA256 "${_trx_archive_path}" _trx_actual_hash)
            if(NOT _trx_actual_hash STREQUAL _trx_expected_hash)
                message(STATUS "Hash mismatch for ${_trx_archive}; re-downloading.")
                file(REMOVE "${_trx_archive_path}")
            endif()
        endif()

        if(NOT EXISTS "${_trx_archive_path}")
            message(STATUS "Downloading ${_trx_archive}...")
            file(DOWNLOAD "${_trx_archive_url}" "${_trx_archive_path}"
                SHOW_PROGRESS
                TLS_VERIFY ON
                EXPECTED_HASH "SHA256=${_trx_expected_hash}"
                STATUS _download_status
            )
            list(GET _download_status 0 _status_code)
            if(NOT _status_code EQUAL 0)
                list(GET _download_status 1 _status_message)
                message(FATAL_ERROR "Failed to download ${_trx_archive}: ${_status_message}")
            endif()
        endif()

        if(NOT EXISTS "${_trx_marker}")
            message(STATUS "Extracting ${_trx_archive}...")
            file(ARCHIVE_EXTRACT INPUT "${_trx_archive_path}" DESTINATION "${TRX_TEST_DATA_DIR}")
            file(TOUCH "${_trx_marker}")
        endif()
    endforeach()
    message(STATUS "Test data available at ${TRX_TEST_DATA_DIR}")
else()
    message(STATUS "TRX_DOWNLOAD_TEST_DATA=OFF; skipping test data download.")
endif()

add_executable(test_mmap test_trx_mmap.cpp)
target_link_libraries(test_mmap PRIVATE trx GTest::gtest_main)
target_compile_features(test_mmap PRIVATE cxx_std_17)

add_executable(test_io test_trx_io.cpp)
target_link_libraries(test_io PRIVATE trx GTest::gtest_main)
target_compile_features(test_io PRIVATE cxx_std_17)

add_executable(test_streamlines_ops test_trx_streamlines_ops.cpp)
target_link_libraries(test_streamlines_ops PRIVATE trx GTest::gtest_main)
target_compile_features(test_streamlines_ops PRIVATE cxx_std_17)

add_executable(test_filesystem test_trx_filesystem.cpp)
target_link_libraries(test_filesystem PRIVATE trx GTest::gtest_main)
target_compile_features(test_filesystem PRIVATE cxx_std_17)

add_executable(test_trxfile test_trx_trxfile.cpp)
target_link_libraries(test_trxfile PRIVATE trx GTest::gtest_main)
target_compile_features(test_trxfile PRIVATE cxx_std_17)

add_executable(test_anytrxfile test_trx_anytrxfile.cpp)
target_link_libraries(test_anytrxfile PRIVATE trx GTest::gtest_main)
target_compile_features(test_anytrxfile PRIVATE cxx_std_17)

include(GoogleTest)
gtest_discover_tests(test_mmap PROPERTIES
    ENVIRONMENT "TRX_TEST_DATA_DIR=${TRX_TEST_DATA_DIR}"
)

gtest_discover_tests(test_io PROPERTIES
    ENVIRONMENT "TRX_TEST_DATA_DIR=${TRX_TEST_DATA_DIR}"
)

gtest_discover_tests(test_streamlines_ops)

gtest_discover_tests(test_filesystem)

gtest_discover_tests(test_trxfile PROPERTIES
    ENVIRONMENT "TRX_TEST_DATA_DIR=${TRX_TEST_DATA_DIR}"
)

gtest_discover_tests(test_anytrxfile PROPERTIES
    ENVIRONMENT "TRX_TEST_DATA_DIR=${TRX_TEST_DATA_DIR}"
)

enable_testing()

find_package(GTest CONFIG QUIET)
if(NOT GTest_FOUND)
    find_package(GTest REQUIRED)
endif()

add_executable(test_mmap test_trx_mmap.cpp)
target_link_libraries(test_mmap PRIVATE trx GTest::gtest GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(test_mmap)

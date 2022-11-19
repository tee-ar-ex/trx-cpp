include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
include_directories(${PROJECT_SOURCE_DIR}/src)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_executable(tests src/test_trx_mmap.cpp)
target_link_libraries(tests PRIVATE ${CONAN_LIBS} ${LOCAL_PROJECT_NAME})

gtest_discover_tests(tests)

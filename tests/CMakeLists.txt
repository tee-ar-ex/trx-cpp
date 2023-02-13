if(TRX_USE_CONAN)
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
endif()

include_directories(${PROJECT_SOURCE_DIR}/src)

enable_testing()
find_package(GTest REQUIRED)
include(GoogleTest)

add_definitions(-DDATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

add_executable(tests src/test_trx_mmap.cpp)
if(TRX_USE_CONAN)
    target_link_libraries(tests PRIVATE ${CONAN_LIBS} ${LOCAL_PROJECT_NAME})
else()
    target_link_libraries(tests PRIVATE GTest::GTest GTest::Main)
endif()

gtest_discover_tests(tests)

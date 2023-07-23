set(LOCAL_PROJECT_NAME "trx")
set(LOCAL_PROJECT_VERSION "0.1.0")

cmake_minimum_required(VERSION 3.0.0)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)
project(${LOCAL_PROJECT_NAME} VERSION ${LOCAL_PROJECT_VERSION})
set(CMAKE_CXX_STANDARD 11)

option(TRX_USE_CONAN "Should Conan package manager be used?" ON)

#set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

add_library(${LOCAL_PROJECT_NAME} src/trx.cpp)

include(cmake/ConanSetup.cmake)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

option(TRX_TESTS "Build the tests" ON)
if(TRX_TESTS)
    add_subdirectory(tests)
endif()

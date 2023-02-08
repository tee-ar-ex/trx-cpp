cmake_minimum_required(VERSION 3.0.0)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)
project(trx_example)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

add_executable(load_trx load_trx.cpp)

include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
conan_basic_setup()

# target_link_libraries(
#     ${LOCAL_PROJECT_NAME}
#     PRIVATE ${CONAN_LIBS}
# )

# target_include_directories(${LOCAL_PROJECT_NAME}
#     PUBLIC
#         $<INSTALL_INTERFACE:include>
#         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
#     PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/src
#         ${CMAKE_CURRENT_BINARY_DIR}/gen-private-include
#         ${CONAN_INCLUDE_DIRS}
# )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

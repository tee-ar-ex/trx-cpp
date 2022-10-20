cmake_minimum_required(VERSION 3.0.0)
cmake_policy(SET CMP0074 NEW)
cmake_policy(SET CMP0079 NEW)
project(trx)
set (CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Debug)

find_package(libzip REQUIRED)
find_package (Eigen3 CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

include_directories(../src)
add_executable(load_trx load_trx.cpp ../src/trx.h ../src/trx.tpp ../src/trx.cpp)


TARGET_LINK_LIBRARIES(load_trx
	nlohmann_json::nlohmann_json
	Eigen3::Eigen
	libzip::zip
	spdlog::spdlog
	spdlog::spdlog_header_only
)


set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})

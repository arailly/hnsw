cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_STANDARD 14)
project(hnsw)

add_executable(hnsw main.cpp)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp -O3")

include_directories(${PROJECT_SOURCE_DIR} ${PROJECT_SOURCE_DIR}/include)

add_subdirectory(test)
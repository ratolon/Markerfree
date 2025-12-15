cmake_minimum_required(VERSION 3.18)

project(markerfree LANGUAGES C CXX CUDA)

# Declare GPU architectures in a modern way (CMake >= 3.18)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 62 70 75 80 86 89 90)

include_directories(${MSGMAP_INCLUDE_DIR} ${PROJECT_SOURCE_DIR}/src)
link_directories (${PROJECT_BINARY_DIR}/lib)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR}/bin)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

add_subdirectory(src)

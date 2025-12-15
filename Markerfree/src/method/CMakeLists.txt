cmake_minimum_required(VERSION 3.18)

file(GLOB METHOD_SOURCES "*.cpp" "*.cu" "*.h" "*.cuh")
message(STATUS "METHOD_SOURCES: ${METHOD_SOURCES}")

if(ENABLE_OMP)
       find_package(OpenMP REQUIRED)
       set(CMAKE_C_FLAGS "${OpenMP_C_FLAGS} ${CMAKE_C_FLAGS}")
       set(CMAKE_CXX_FLAGS "${OpenMP_CXX_FLAGS} ${CMAKE_C_FLAGS}")
endif(ENABLE_OMP)

find_package(MPI REQUIRED COMPONENTS C CXX)
find_package(CUDAToolkit REQUIRED)

include_directories(${PROJECT_SOURCE_DIR})

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_CUDA_ARCHITECTURES 52 60 61 62 70 75 80 86 89 90)

add_library(method_library STATIC ${METHOD_SOURCES})

# CUDA-specific compile options (only applied to CUDA sources)
target_compile_options(method_library PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)

target_include_directories(method_library PRIVATE ${PROJECT_SOURCE_DIR})

target_link_libraries(method_library PRIVATE
    MPI::MPI_CXX
    CUDA::cufft
    CUDA::cudart
    mrc)

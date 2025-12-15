cmake_minimum_required(VERSION 3.18)
cmake_policy(SET CMP0146 NEW)  # 解决 FindCUDA 警告

# 启用 CUDA 并自动查找 CUDA 工具包
enable_language(CUDA)
find_package(CUDAToolkit REQUIRED)

# 其他配置（MPI、OpenMP 等）
find_package(MPI REQUIRED COMPONENTS C CXX)
if(ENABLE_OMP)
    find_package(OpenMP REQUIRED)
endif()

# 源文件
FILE(GLOB SOURCES "*.cu" "*.cpp" "*.c" "*.h" "*.cuh")

# 创建可执行文件
add_executable(Markerfree ${SOURCES})

# 链接库
target_link_libraries(Markerfree
    CUDA::cufft      # 现代 CMake 的 CUDA 目标
    CUDA::cudart     # CUDA 运行时库
    MPI::MPI_CXX
    method_library
    mrc
    ${OpenMP_CXX_LIBRARIES}
)

# 包含目录
target_include_directories(Markerfree PRIVATE
    ${MPI_INCLUDE_PATH}
)

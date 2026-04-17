cmake_minimum_required(VERSION 4.2)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(leetgpu CUDA)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# Configure Debug vs Release flags
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Setting CUDA Debug flags")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G -O0")
else()
    set(CMAKE_BUILD_TYPE "Release")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

add_executable(01_reduction 01_reduction/01_reduction.cu)
add_executable(02_softmax 02_softmax/02_softmax.cu)

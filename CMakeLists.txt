cmake_minimum_required(VERSION 4.2)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
project(leetgpu CUDA)

set(CMAKE_CUDA_STANDARD 20)

add_executable(01_reduction 01_reduction/01_reduction.cu)

set_target_properties(01_reduction PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON)

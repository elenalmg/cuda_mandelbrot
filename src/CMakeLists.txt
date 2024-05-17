cmake_minimum_required(VERSION 3.12)
project(MandelbrotProject LANGUAGES CXX CUDA)

find_package(Python3 COMPONENTS Development REQUIRED)
find_package(pybind11 REQUIRED)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Specify CUDA architecture
set(CMAKE_CUDA_ARCHITECTURES 75)

add_library(mandelbrot_module SHARED
    mandelbrot_pybind.cpp
    mandelbrot.cpp
    mandelbrot_cuda.cu
)

set_target_properties(mandelbrot_module PROPERTIES
    PREFIX "${PYTHON_PREFIX}"
    SUFFIX ".so"
)

target_include_directories(mandelbrot_module PRIVATE
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

target_link_libraries(mandelbrot_module PRIVATE
    ${Python3_LIBRARIES}
    CUDA::cudart
)

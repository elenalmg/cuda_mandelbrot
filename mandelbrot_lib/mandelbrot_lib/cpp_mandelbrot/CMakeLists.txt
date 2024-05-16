cmake_minimum_required(VERSION 3.12)
project(mandelbrot_cpp)

set(CMAKE_CXX_STANDARD 11)

find_package(pybind11 REQUIRED)

pybind11_add_module(mandelbrot_cpp cpp_algo.cpp)

add_library(cuda_mandelbrot_lib SHARED
    pybind.cpp
    multithread.cpp
    base_cuda.cu
    manual_unroll.cu
    polynomial_unroll.cu
    n_unroll.cu
)

target_include_directories(cuda_mandelbrot_lib PUBLIC ${CMAKE_SOURCE_DIR}/pybind11/include)
target_link_libraries(cuda_mandelbrot_lib PUBLIC
  Python3::Python
  cudart)

set_target_properties(cuda_mandelbrot_lib PROPERTIES PREFIX "")

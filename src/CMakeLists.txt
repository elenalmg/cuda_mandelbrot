add_library(cuda_mandelbrot_lib SHARED
  mandelbrot_pybind.cpp)

#target_include_directories(cuda_mandelbrot_lib PUBLIC ${Python3_INCLUDE_DIRS})
target_include_directories(cuda_mandelbrot_lib PUBLIC ${CMAKE_SOURCE_DIR}/pybind11/include)
target_link_libraries(cuda_mandelbrot_lib PUBLIC
  #${PYTHON3_PYTHON}
  Python3::Python
  cudart)

# set what comes before the library name, here no "lib"
set_target_properties(cuda_mandelbrot_lib PROPERTIES PREFIX "")

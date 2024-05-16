cmake_minimum_required(VERSION 3.12)
project(CudaMandelbrot)
find_package(CUDA REQUIRED)
find_package(PythonInterp 3.12 REQUIRED)

# Ajoutez les chemins trouvés ici (à modifier)
set(PYTHON_INCLUDE_DIRS -I/opt/mamba/include/python3.12)
set(PYTHON_LIBRARIES -L/opt/mamba/lib/python3.12/config-3.12-x86_64-linux-gnu)

# marche pas
include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ./pybind11/include
    )

link_directories(
    /usr/local/cuda/lib64
    )

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

cuda_add_library(cuda_mandelbrot SHARED
  cuda.cu)

target_link_libraries(cuda_mandelbrot
  ${PYTHON_LIBRARIES}
  cudart)

set_target_properties(cuda_mandelbrot PROPERTIES PREFIX "")

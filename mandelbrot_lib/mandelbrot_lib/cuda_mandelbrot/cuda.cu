#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <iostream>
#include <cuda_runtime.h>
#include "cuComplex.h"

namespace py = pybind11;

// Sript qui fonctionne, mais je n'arrive pas à le transformer en module python
// obliger de mettre dans un dossier commun avec le CMakeLists car sinon cela ne trouve pas le script

// ou dans un autre fichier ?
__device__ int _compute_escape_iter(float x, float y, int max_iter, float escape_radius) {
    cuFloatComplex c = make_cuFloatComplex(x, y);
    cuFloatComplex z = make_cuFloatComplex(0.0f, 0.0f);
    for (int i = 0; i < max_iter; i++) {
        z = cuCaddf(cuCmulf(z, z), c);
        if (cuCrealf(z) * cuCrealf(z) + cuCimagf(z) * cuCimagf(z) >= escape_radius * escape_radius) {
            return i;
        }
    }
    return -1;
}

__global__ void compute_grid_kernel(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, int* results, float escape_radius) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    float pixel_size_x = (x_max - x_min) / width;
    float pixel_size_y = (y_max - y_min) / height;
    if (x < width && y < height) {
        float real = x_min + x * pixel_size_x;
        float imag = y_min + y * pixel_size_y;
        results[y * width + x] = _compute_escape_iter(real, imag, max_iter, escape_radius);
    }
}


class MandelbrotCuda {
public:
    float escape_radius;

    MandelbrotCuda(float escape_radius) : escape_radius(escape_radius) {}

    __device__ int compute_escape_iter(float x, float y, int max_iter) {
        return _compute_escape_iter(x, y, max_iter, escape_radius);
    }

    void compute_grid(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter) {
        int* results;
        cudaMallocManaged(&results, width * height * sizeof(int));

        dim3 block_size(16, 16);
        dim3 grid_size((width + block_size.x - 1) / block_size.x, (height + block_size.y - 1) / block_size.y);
        compute_grid_kernel<<<grid_size, block_size>>>(x_min, y_min, x_max, y_max, width, height, max_iter, results, escape_radius);

        cudaDeviceSynchronize();
        
        cudaFree(results);
    }
};


void mandelbrot_cuda(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius, py::array_t<int> results) {
    int* results_ptr = static_cast<int*>(results.request().ptr);

    MandelbrotCuda mandelbrot(escape_radius);
    mandelbrot.compute_grid(x_min, y_min, x_max, y_max, width, height, max_iter, results_ptr);
}

PYBIND11_MODULE(mandelbrot, m) {
    m.def("mandelbrot_cuda", &mandelbrot_cuda, "Compute Mandelbrot set using CUDA",
          py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"),
          py::arg("width"), py::arg("height"), py::arg("max_iter"), py::arg("escape_radius"),
          py::arg("results"));
}
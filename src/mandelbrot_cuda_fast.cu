#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>

__global__ void mandelbrot_kernel_fast(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < height) {
        float pixel_size_x = (x_max - x_min) / width;
        float pixel_size_y = (y_max - y_min) / height;
        float real = x_min + idx * pixel_size_x;
        float imag = y_min + idy * pixel_size_y;
        float zr = real, zi = imag;
        float zr2 = zr * zr, zi2 = zi * zi;
        int iter = 0;

        while (zr2 + zi2 < escape_radius_squared && iter < max_iter) {
            float new_zr = zr2 - zi2 + real;
            float new_zi = 2.0f * zr * zi + imag;

            zr = new_zr * new_zr - new_zi * new_zi + real;
            zi = 2.0f * new_zr * new_zi + imag;

            zr2 = zr * zr;
            zi2 = zi * zi;

            iter += 2; // Two iterations per loop
        }

        if (zr2 + zi2 >= escape_radius_squared) {
            results[idy * width + idx] = iter;
        } else {
            results[idy * width + idx] = max_iter;
        }
    }
}

extern "C" void compute_grid_cuda_fast(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, int* results) {
    int* d_results;
    size_t size = width * height * sizeof(int);
    cudaMalloc(&d_results, size);
    
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);
    
    mandelbrot_kernel_fast<<<num_blocks, threads_per_block>>>(x_min, y_min, x_max, y_max, width, height, max_iter, escape_radius_squared, d_results);
    
    cudaMemcpy(results, d_results, size, cudaMemcpyDeviceToHost);
    cudaFree(d_results);
}


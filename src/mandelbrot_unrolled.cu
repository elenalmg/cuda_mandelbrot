#include <vector>
#include <cuda_runtime.h>

__device__ int mandelbrot_unroll_2(float x, float y, float escape_radius_squared) {
    float zr = x, zi = y;
    float zr2 = zr * zr, zi2 = zi * zi;

    // Hard-coded to be able to unroll at compile time
    int unroll_factor = 2;
    int override_max_iter = 200;

    #pragma unroll
    for (int i = 0; i < override_max_iter; i += unroll_factor) {
        #pragma unroll
        for (int j = 0; j < unroll_factor; ++j) {
            // Compute the new zr and zi
            float zr_new = zr2 - zi2 + x;
            float zi_new = 2.0f * zr * zi + y;

            zr = zr_new;
            zi = zi_new;
            zr2 = zr * zr;
            zi2 = zi * zi;

            // If escape condition is met, return the current iteration count
            if (zr2 + zi2 >= escape_radius_squared) {
                return i + j + 1; // i + j because we need to account for the inner loop iterations
            }
        }
    }

    return max_iter;
}

__global__ void mandelbrot_kernel_2(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < height) {
        float pixel_size_x = (x_max - x_min) / width;
        float pixel_size_y = (y_max - y_min) / height;
        float real = x_min + idx * pixel_size_x;
        float imag = y_min + idy * pixel_size_y;

        results[idy * width + idx] = mandelbrot_unroll_2(real, imag, escape_radius_squared);
    }
}

extern "C" void compute_grid_cuda_2(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results) {
    int* d_results;
    size_t size = width * height * sizeof(int);
    cudaMalloc(&d_results, size);
    
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);

    mandelbrot_kernel_2<<<num_blocks, threads_per_block>>>(x_min, y_min, x_max, y_max, width, height, escape_radius_squared, d_results);

    cudaMemcpy(results, d_results, size, cudaMemcpyDeviceToHost);
    cudaFree(d_results);
}

// 3
__device__ int mandelbrot_unroll_3(float x, float y, float escape_radius_squared) {
    float zr = x, zi = y;
    float zr2 = zr * zr, zi2 = zi * zi;

    // Hard-coded to be able to unroll at compile time
    int unroll_factor = 3;
    int override_max_iter = 200;

    #pragma unroll
    for (int i = 0; i < override_max_iter; i += unroll_factor) {
        #pragma unroll
        for (int j = 0; j < unroll_factor; ++j) {
            // Compute the new zr and zi
            float zr_new = zr2 - zi2 + x;
            float zi_new = 2.0f * zr * zi + y;

            zr = zr_new;
            zi = zi_new;
            zr2 = zr * zr;
            zi2 = zi * zi;

            // If escape condition is met, return the current iteration count
            if (zr2 + zi2 >= escape_radius_squared) {
                return i + j + 1; // i + j because we need to account for the inner loop iterations
            }
        }
    }

    return max_iter;
}

__global__ void mandelbrot_kernel_3(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < height) {
        float pixel_size_x = (x_max - x_min) / width;
        float pixel_size_y = (y_max - y_min) / height;
        float real = x_min + idx * pixel_size_x;
        float imag = y_min + idy * pixel_size_y;

        results[idy * width + idx] = mandelbrot_unroll_3(real, imag, escape_radius_squared);
    }
}

extern "C" void compute_grid_cuda_3(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results) {
    int* d_results;
    size_t size = width * height * sizeof(int);
    cudaMalloc(&d_results, size);
    
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);

    mandelbrot_kernel_3<<<num_blocks, threads_per_block>>>(x_min, y_min, x_max, y_max, width, height, escape_radius_squared, d_results);

    cudaMemcpy(results, d_results, size, cudaMemcpyDeviceToHost);
    cudaFree(d_results);
}

// 5
__device__ int mandelbrot_unroll_5(float x, float y, float escape_radius_squared) {
    float zr = x, zi = y;
    float zr2 = zr * zr, zi2 = zi * zi;

    // Hard-coded to be able to unroll at compile time
    int unroll_factor = 5;
    int override_max_iter = 200;

    #pragma unroll
    for (int i = 0; i < override_max_iter; i += unroll_factor) {
        #pragma unroll
        for (int j = 0; j < unroll_factor; ++j) {
            // Compute the new zr and zi
            float zr_new = zr2 - zi2 + x;
            float zi_new = 2.0f * zr * zi + y;

            zr = zr_new;
            zi = zi_new;
            zr2 = zr * zr;
            zi2 = zi * zi;

            // If escape condition is met, return the current iteration count
            if (zr2 + zi2 >= escape_radius_squared) {
                return i + j + 1; // i + j because we need to account for the inner loop iterations
            }
        }
    }

    return max_iter;
}

__global__ void mandelbrot_kernel_5(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < height) {
        float pixel_size_x = (x_max - x_min) / width;
        float pixel_size_y = (y_max - y_min) / height;
        float real = x_min + idx * pixel_size_x;
        float imag = y_min + idy * pixel_size_y;

        results[idy * width + idx] = mandelbrot_unroll_5(real, imag, escape_radius_squared);
    }
}

extern "C" void compute_grid_cuda_5(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results) {
    int* d_results;
    size_t size = width * height * sizeof(int);
    cudaMalloc(&d_results, size);
    
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);

    mandelbrot_kernel_5<<<num_blocks, threads_per_block>>>(x_min, y_min, x_max, y_max, width, height, escape_radius_squared, d_results);

    cudaMemcpy(results, d_results, size, cudaMemcpyDeviceToHost);
    cudaFree(d_results);
}

// 10
__device__ int mandelbrot_unroll_10(float x, float y, float escape_radius_squared) {
    float zr = x, zi = y;
    float zr2 = zr * zr, zi2 = zi * zi;

    // Hard-coded to be able to unroll at compile time
    int unroll_factor = 10;
    int override_max_iter = 200;

    #pragma unroll
    for (int i = 0; i < override_max_iter; i += unroll_factor) {
        #pragma unroll
        for (int j = 0; j < unroll_factor; ++j) {
            // Compute the new zr and zi
            float zr_new = zr2 - zi2 + x;
            float zi_new = 2.0f * zr * zi + y;

            zr = zr_new;
            zi = zi_new;
            zr2 = zr * zr;
            zi2 = zi * zi;

            // If escape condition is met, return the current iteration count
            if (zr2 + zi2 >= escape_radius_squared) {
                return i + j + 1; // i + j because we need to account for the inner loop iterations
            }
        }
    }

    return max_iter;
}

__global__ void mandelbrot_kernel_10(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < height) {
        float pixel_size_x = (x_max - x_min) / width;
        float pixel_size_y = (y_max - y_min) / height;
        float real = x_min + idx * pixel_size_x;
        float imag = y_min + idy * pixel_size_y;

        results[idy * width + idx] = mandelbrot_unroll_10(real, imag, escape_radius_squared);
    }
}

extern "C" void compute_grid_cuda_10(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results) {
    int* d_results;
    size_t size = width * height * sizeof(int);
    cudaMalloc(&d_results, size);
    
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);

    mandelbrot_kernel_10<<<num_blocks, threads_per_block>>>(x_min, y_min, x_max, y_max, width, height, escape_radius_squared, d_results);

    cudaMemcpy(results, d_results, size, cudaMemcpyDeviceToHost);
    cudaFree(d_results);
}
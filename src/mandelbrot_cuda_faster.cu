#include <vector>
#include <cuda_runtime.h>


__global__ void mandelbrot_kernel_faster(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < height) {
        float pixel_size_x = (x_max - x_min) / width;
        float pixel_size_y = (y_max - y_min) / height;
        float real = x_min + idx * pixel_size_x;
        float imag = y_min + idy * pixel_size_y;

        float zr = real, zi = imag;
        float zr2 = zr * zr, zi2 = zi * zi;

        for (int i = 0; i < max_iter; i += 2) {
            if (zr2 + zi2 >= escape_radius_squared) {
                results[idy * width + idx] = i;
                return;
            }

            // Compute intermediate terms for the polynomial
            float zr4 = zr2 * zr2;
            float zi4 = zi2 * zi2;
            float zrzi = zr * zi;
            float zr2zi2 = zr2 * zi2;
            float zr3zi = zrzi * zr2;
            float zrzi3 = zrzi * zi2;
            float zr2_real = zr2 * real;
            float zi2_real = zi2 * real;
            float zrzi_imag = zrzi * imag;
            float zrzi_real = zrzi * real;

            // Calculate new zr and zi using the polynomial
            float new_zr = zr4 - 6.0f * zr2zi2 + zi4 + 2.0f * zr2_real - 2.0f * zi2_real + real * real - 4.0f * zrzi_imag - imag * imag + real;
            float new_zi = 4.0f * zr3zi + 2.0f * zr2 * imag - 4.0f * zrzi3 - 2.0f * zi2 * imag + 4.0f * zrzi_real + 2.0f * real * imag + imag;

            zr = new_zr;
            zi = new_zi;
            zr2 = zr * zr;
            zi2 = zi * zi;

            if (zr2 + zi2 >= escape_radius_squared) {
                results[idy * width + idx] = i + 1;
                return;
            }
        }
        results[idy * width + idx] = max_iter;
    }
}

extern "C" void compute_grid_cuda_faster(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, int* results) {
    int* d_results;
    size_t size = width * height * sizeof(int);
    cudaMalloc(&d_results, size);
    
    dim3 threads_per_block(16, 16);
    dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);

    mandelbrot_kernel_faster<<<num_blocks, threads_per_block>>>(x_min, y_min, x_max, y_max, width, height, max_iter, escape_radius_squared, d_results);

    cudaMemcpy(results, d_results, size, cudaMemcpyDeviceToHost);
    cudaFree(d_results);
}

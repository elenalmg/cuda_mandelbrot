#include <vector>
#include <cuda_runtime.h>

__global__ void mandelbrot_kernel(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, int* results) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if (idx < width && idy < height) {
        float pixel_size_x = (x_max - x_min) / width;
        float pixel_size_y = (y_max - y_min) / height;
        float real = x_min + idx * pixel_size_x;
        float imag = y_min + idy * pixel_size_y;
        float zr = 0.0f, zi = 0.0f;
        float zr2 = 0.0f, zi2 = 0.0f;
        int iter = 0;
        while (zr2 + zi2 < escape_radius_squared && iter < max_iter) {
            zi = 2.0f * zr * zi + imag;
            zr = zr2 - zi2 + real;
            zr2 = zr * zr;
            zi2 = zi * zi;
            iter++;
        }
        results[idy * width + idx] = iter;
    }
}

class MandelbrotCUDA {
public:
    MandelbrotCUDA(float escape_radius) : escape_radius_squared(escape_radius * escape_radius) {}

    void compute_grid(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, std::vector<int>& results) const {
        int* d_results;
        size_t size = width * height * sizeof(int);
        results.resize(width * height);

        cudaMalloc(&d_results, size);
        dim3 threads_per_block(16, 16);
        dim3 num_blocks((width + threads_per_block.x - 1) / threads_per_block.x, (height + threads_per_block.y - 1) / threads_per_block.y);
        
        mandelbrot_kernel<<<num_blocks, threads_per_block>>>(x_min, y_min, x_max, y_max, width, height, max_iter, escape_radius_squared, d_results);
        cudaMemcpy(results.data(), d_results, size, cudaMemcpyDeviceToHost);
        cudaFree(d_results);
    }

private:
    float escape_radius_squared;
};

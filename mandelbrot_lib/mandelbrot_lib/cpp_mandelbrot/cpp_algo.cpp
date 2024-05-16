#include <complex>
#include <vector>
#include "cuda_mandelbrot/mandelbrot_lib/mandelbrot_lib/cpp_mandelbrot/cpp_algo.h" 

// Script Fonctionne normalement

class MandelbrotCPP {
public:
    MandelbrotCPP(float escape_radius) : escape_radius(escape_radius) {}

    int compute_escape_iter(float x, float y, int max_iter) const {
        std::complex<float> c(x, y);
        std::complex<float> z(0.0f, 0.0f);
        for (int i = 0; i < max_iter; i++) {
            z = z * z + c;
            if (std::norm(z) >= escape_radius * escape_radius) {
                return i;
            }
        }
        return -1;
    }

    void compute_grid(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, std::vector<int>& results) const {
        float pixel_size_x = (x_max - x_min) / width;
        float pixel_size_y = (y_max - y_min) / height;
        results.resize(width * height);
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float real = x_min + x * pixel_size_x;
                float imag = y_min + y * pixel_size_y;
                results[y * width + x] = compute_escape_iter(real, imag, max_iter);
            }
        }
    }

private:
    float escape_radius;
};


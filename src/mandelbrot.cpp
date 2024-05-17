#include <vector>
#include <complex>
#include <thread>

class MandelbrotCPP {
public:
    MandelbrotCPP(float escape_radius) : escape_radius_squared(escape_radius * escape_radius) {}

    int compute_escape_iter(float x, float y, int max_iter) const {
        float zr = 0.0f, zi = 0.0f;
        float zr2 = 0.0f, zi2 = 0.0f;
        for (int i = 0; i < max_iter; i++) {
            zi = 2.0f * zr * zi + y;
            zr = zr2 - zi2 + x;
            zr2 = zr * zr;
            zi2 = zi * zi;
            if (zr2 + zi2 >= escape_radius_squared) {
                return i;
            }
        }
        return max_iter;
    }

    void compute_grid(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, std::vector<int>& results) const {
        float pixel_size_x = (x_max - x_min) / width;
        float pixel_size_y = (y_max - y_min) / height;
        results.resize(width * height);

        auto compute_row = [&](int start, int end) {
            for (int y = start; y < end; ++y) {
                for (int x = 0; x < width; ++x) {
                    float real = x_min + x * pixel_size_x;
                    float imag = y_min + y * pixel_size_y;
                    results[y * width + x] = compute_escape_iter(real, imag, max_iter);
                }
            }
        };

        int num_threads = std::thread::hardware_concurrency();
        std::vector<std::thread> threads(num_threads);
        int rows_per_thread = height / num_threads;

        for (int i = 0; i < num_threads; ++i) {
            int start = i * rows_per_thread;
            int end = (i == num_threads - 1) ? height : start + rows_per_thread;
            threads[i] = std::thread(compute_row, start, end);
        }

        for (auto& t : threads) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

private:
    float escape_radius_squared;
};

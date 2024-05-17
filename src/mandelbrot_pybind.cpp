#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mandelbrot.cpp"

extern "C" void compute_grid_cuda(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, int* results);

namespace py = pybind11;

class MandelbrotCUDA {
public:
    MandelbrotCUDA(float escape_radius) : escape_radius_squared(escape_radius * escape_radius) {}

    void compute_grid(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, std::vector<int>& results) const {
        results.resize(width * height);
        compute_grid_cuda(x_min, y_min, x_max, y_max, width, height, max_iter, escape_radius_squared, results.data());
    }

private:
    float escape_radius_squared;
};

PYBIND11_MODULE(cuda_mandelbrot_lib, m) {
    py::class_<MandelbrotCPP>(m, "CPPMandelbrot")
        .def(py::init<float>())
        .def("compute_escape_iter", &MandelbrotCPP::compute_escape_iter)
        .def("compute_grid", &MandelbrotCPP::compute_grid);

    py::class_<MandelbrotCUDA>(m, "CUDAMandelbrot")
        .def(py::init<float>())
        .def("compute_grid", &MandelbrotCUDA::compute_grid);
}

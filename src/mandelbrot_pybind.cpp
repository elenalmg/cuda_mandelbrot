#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mandelbrot.cpp"


namespace py = pybind11;

extern "C" void compute_grid_cuda(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, int* results);
extern "C" void compute_grid_cuda_smooth(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, float* results);

extern "C" void compute_grid_cuda_fast(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, int* results);
extern "C" void compute_grid_cuda_smooth_fast(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, float* results);

class MandelbrotCUDA {
public:
    MandelbrotCUDA(float escape_radius) : escape_radius_squared(escape_radius * escape_radius) {}

    pybind11::array_t<int> compute_grid(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter) const {
        auto result = pybind11::array_t<int>(width * height);
        auto buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);

        compute_grid_cuda(x_min, y_min, x_max, y_max, width, height, max_iter, escape_radius_squared, ptr);

        result.resize({height, width});
        return result;
    }

    pybind11::array_t<float> compute_grid_smooth(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter) const {
        auto result = pybind11::array_t<float>(width * height);
        auto buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);

        compute_grid_cuda_smooth(x_min, y_min, x_max, y_max, width, height, max_iter, escape_radius_squared, ptr);

        result.resize({height, width});
        return result;
    }

private:
    float escape_radius_squared;
};



class FastMandelbrotCUDA {
public:
    FastMandelbrotCUDA(float escape_radius) : escape_radius_squared(escape_radius * escape_radius) {}

    pybind11::array_t<int> compute_grid(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter) const {
        auto result = pybind11::array_t<int>(width * height);
        auto buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);

        compute_grid_cuda_fast(x_min, y_min, x_max, y_max, width, height, max_iter, escape_radius_squared, ptr);

        result.resize({height, width});
        return result;
    }

    pybind11::array_t<float> compute_grid_smooth(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter) const {
        auto result = pybind11::array_t<float>(width * height);
        auto buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);

        compute_grid_cuda_smooth_fast(x_min, y_min, x_max, y_max, width, height, max_iter, escape_radius_squared, ptr);

        result.resize({height, width});
        return result;
    }

private:
    float escape_radius_squared;
};

PYBIND11_MODULE(cuda_mandelbrot_lib, m) {
    py::class_<MandelbrotCPP>(m, "MandelbrotCPP")
        .def(py::init<float>(), py::arg("escape_radius"))
        .def("compute_escape_iter", &MandelbrotCPP::compute_escape_iter, py::arg("x"), py::arg("y"), py::arg("max_iter"))
        .def("compute_grid", &MandelbrotCPP::compute_grid, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"), py::arg("max_iter"));

    py::class_<MandelbrotCUDA>(m, "MandelbrotCUDA")
        .def(py::init<float>(), py::arg("escape_radius"))
        .def("compute_grid", &MandelbrotCUDA::compute_grid, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"), py::arg("max_iter"))
        .def("compute_grid_smooth", &MandelbrotCUDA::compute_grid_smooth, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"), py::arg("max_iter"));
    py::class_<FastMandelbrotCUDA>(m, "FastMandelbrotCUDA")
        .def(py::init<float>(), py::arg("escape_radius"))
        .def("compute_grid", &FastMandelbrotCUDA::compute_grid, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"), py::arg("max_iter"))
        .def("compute_grid_smooth", &FastMandelbrotCUDA::compute_grid_smooth, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"), py::arg("max_iter"));
}

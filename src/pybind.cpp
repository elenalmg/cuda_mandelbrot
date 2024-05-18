#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "multithread.cpp"


namespace py = pybind11;

extern "C" void compute_grid_cuda(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, int* results);
extern "C" void compute_grid_cuda_smooth(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, float* results);

extern "C" void compute_grid_cuda_manual_unroll(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, int* results);
extern "C" void compute_grid_cuda_smooth_manual_unroll(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, float* results);

extern "C" void compute_grid_cuda_math_unroll(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter, float escape_radius_squared, int* results);

extern "C" void compute_grid_cuda_2(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results);
extern "C" void compute_grid_cuda_3(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results);
extern "C" void compute_grid_cuda_5(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results);
extern "C" void compute_grid_cuda_10(float x_min, float y_min, float x_max, float y_max, int width, int height, float escape_radius_squared, int* results);


class BaseCUDA {
public:
    BaseCUDA(float escape_radius) : escape_radius_squared(escape_radius * escape_radius) {}

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



class ManualUnroll {
public:
    ManualUnroll(float escape_radius) : escape_radius_squared(escape_radius * escape_radius) {}

    pybind11::array_t<int> compute_grid(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter) const {
        auto result = pybind11::array_t<int>(width * height);
        auto buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);

        compute_grid_cuda_manual_unroll(x_min, y_min, x_max, y_max, width, height, max_iter, escape_radius_squared, ptr);

        result.resize({height, width});
        return result;
    }

    pybind11::array_t<float> compute_grid_smooth(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter) const {
        auto result = pybind11::array_t<float>(width * height);
        auto buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);

        compute_grid_cuda_smooth_manual_unroll(x_min, y_min, x_max, y_max, width, height, max_iter, escape_radius_squared, ptr);

        result.resize({height, width});
        return result;
    }

private:
    float escape_radius_squared;
};

class PolynomialUnroll {
public:
    PolynomialUnroll(float escape_radius) : escape_radius_squared(escape_radius * escape_radius) {}

    pybind11::array_t<int> compute_grid(float x_min, float y_min, float x_max, float y_max, int width, int height, int max_iter) const {
        auto result = pybind11::array_t<int>(width * height);
        auto buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);

        compute_grid_cuda_math_unroll, y_min, x_max, y_max, width, height, max_iter, escape_radius_squared, ptr);

        result.resize({height, width});
        return result;
    }

private:
    float escape_radius_squared;
};

class PragmaUnroll {
public:
    PragmaUnroll(float escape_radius) : escape_radius_squared(escape_radius * escape_radius) {}

    pybind11::array_t<int> compute_grid_2(float x_min, float y_min, float x_max, float y_max, int width, int height) const {
        auto result = pybind11::array_t<int>(width * height);
        auto buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);

        compute_grid_cuda_2(x_min, y_min, x_max, y_max, width, height, escape_radius_squared, ptr);

        result.resize({height, width});
        return result;
    }

    pybind11::array_t<int> compute_grid_3(float x_min, float y_min, float x_max, float y_max, int width, int height) const {
        auto result = pybind11::array_t<int>(width * height);
        auto buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);

        compute_grid_cuda_3(x_min, y_min, x_max, y_max, width, height, escape_radius_squared, ptr);

        result.resize({height, width});
        return result;
    }

    pybind11::array_t<int> compute_grid_5(float x_min, float y_min, float x_max, float y_max, int width, int height) const {
        auto result = pybind11::array_t<int>(width * height);
        auto buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);

        compute_grid_cuda_5(x_min, y_min, x_max, y_max, width, height, escape_radius_squared, ptr);

        result.resize({height, width});
        return result;
    }

    pybind11::array_t<int> compute_grid_10(float x_min, float y_min, float x_max, float y_max, int width, int height) const {
        auto result = pybind11::array_t<int>(width * height);
        auto buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);

        compute_grid_cuda_10(x_min, y_min, x_max, y_max, width, height, escape_radius_squared, ptr);

        result.resize({height, width});
        return result;
    }

private:
    float escape_radius_squared;
};

PYBIND11_MODULE(cuda_mandelbrot_lib, m) {
    py::class_<MultithreadCPP>(m, "MultithreadCPP")
        .def(py::init<float>(), py::arg("escape_radius"))
        .def("compute_escape_iter", &MultithreadCPP::compute_escape_iter, py::arg("x"), py::arg("y"), py::arg("max_iter"))
        .def("compute_grid", &MultithreadCPP::compute_grid, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"), py::arg("max_iter"));

    py::class_<BaseCUDA>(m, "BaseCUDA")
        .def(py::init<float>(), py::arg("escape_radius"))
        .def("compute_grid", &BaseCUDA::compute_grid, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"), py::arg("max_iter"))
        .def("compute_grid_smooth", &BaseCUDA::compute_grid_smooth, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"), py::arg("max_iter"));
    py::class_<ManualUnroll>(m, "ManualUnroll")
        .def(py::init<float>(), py::arg("escape_radius"))
        .def("compute_grid", &ManualUnroll::compute_grid, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"), py::arg("max_iter"))
        .def("compute_grid_smooth", &ManualUnroll::compute_grid_smooth, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"), py::arg("max_iter"));
    py::class_<PolynomialUnroll>(m, "math_unroll")
        .def(py::init<float>(), py::arg("escape_radius"))
        .def("compute_grid", &PolynomialUnroll::compute_grid, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"), py::arg("max_iter"));
    py::class_<PragmaUnroll>(m, "PragmaUnroll")
        .def(py::init<float>(), py::arg("escape_radius"))
        .def("compute_grid_2", &PragmaUnroll::compute_grid_2, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"))
        .def("compute_grid_3", &PragmaUnroll::compute_grid_3, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"))
        .def("compute_grid_5", &PragmaUnroll::compute_grid_5, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"))
        .def("compute_grid_10", &PragmaUnroll::compute_grid_10, py::arg("x_min"), py::arg("y_min"), py::arg("x_max"), py::arg("y_max"), py::arg("width"), py::arg("height"));
}

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "cuda_mandelbrot/mandelbrot_lib/mandelbrot_lib/cpp_mandelbrot/cpp_algo.h" 

namespace py = pybind11;

PYBIND11_MODULE(mandelbrot, m) {
    py::class_<MandelbrotCPP>(m, "MandelbrotCPP")
        .def(py::init<float>())
        .def("compute_escape_iter", &MandelbrotCPP::compute_escape_iter)
        .def("compute_grid", &MandelbrotCPP::compute_grid);
}

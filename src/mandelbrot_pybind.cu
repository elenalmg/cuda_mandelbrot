#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "mandelbrot.cpp"
#include "mandelbrot_cuda.cu"

namespace py = pybind11;

PYBIND11_MODULE(cuda_mandelbrot_lib, m) {
    py::class_<MandelbrotCPP>(m, "CPPMandelbrot")
        .def(py::init<float>())
        .def("compute_escape_iter", &MandelbrotCPP::compute_escape_iter)
        .def("compute_grid", &MandelbrotCPP::compute_grid);

    py::class_<MandelbrotCUDA>(m, "CUDAMandelbrot")
        .def(py::init<float>())
        .def("compute_grid", &MandelbrotCUDA::compute_grid);
}

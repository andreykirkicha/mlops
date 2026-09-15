#include <pybind11/pybind11.h>

namespace py = pybind11;

double add(double a, double b) {
    return a + b;
}

PYBIND11_MODULE(_core, module) {
    module.doc() = "Minimal scalar example for the MLOps bindings homework";
    module.def("add", &add, py::arg("a"), py::arg("b"));
}

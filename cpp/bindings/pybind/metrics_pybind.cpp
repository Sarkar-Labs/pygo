#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_metrics(pybind11::module_& m) {
    m.def("metrics_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("metrics_name", []() { return std::string("metrics"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_metrics_abi_version() { return 1u; }
const char* bind_metrics_name() { return "metrics"; }
}
#endif

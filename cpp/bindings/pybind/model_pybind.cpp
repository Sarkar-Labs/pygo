#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_model(pybind11::module_& m) {
    m.def("model_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("model_name", []() { return std::string("model"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_model_abi_version() { return 1u; }
const char* bind_model_name() { return "model"; }
}
#endif

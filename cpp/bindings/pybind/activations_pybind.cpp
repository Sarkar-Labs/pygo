#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_activations(pybind11::module_& m) {
    m.def("activations_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("activations_name", []() { return std::string("activations"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_activations_abi_version() { return 1u; }
const char* bind_activations_name() { return "activations"; }
}
#endif

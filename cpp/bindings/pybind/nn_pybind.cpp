#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_nn(pybind11::module_& m) {
    m.def("nn_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("nn_name", []() { return std::string("nn"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_nn_abi_version() { return 1u; }
const char* bind_nn_name() { return "nn"; }
}
#endif

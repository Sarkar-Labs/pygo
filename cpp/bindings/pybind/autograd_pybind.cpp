#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_autograd(pybind11::module_& m) {
    m.def("autograd_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("autograd_name", []() { return std::string("autograd"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_autograd_abi_version() { return 1u; }
const char* bind_autograd_name() { return "autograd"; }
}
#endif

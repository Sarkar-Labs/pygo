#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_losses(pybind11::module_& m) {
    m.def("losses_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("losses_name", []() { return std::string("losses"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_losses_abi_version() { return 1u; }
const char* bind_losses_name() { return "losses"; }
}
#endif

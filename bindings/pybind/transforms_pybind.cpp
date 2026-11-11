#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_transforms(pybind11::module_& m) {
    m.def("transforms_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("transforms_name", []() { return std::string("transforms"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_transforms_abi_version() { return 1u; }
const char* bind_transforms_name() { return "transforms"; }
}
#endif

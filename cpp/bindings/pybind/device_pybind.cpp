#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_device(pybind11::module_& m) {
    m.def("device_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("device_name", []() { return std::string("device"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_device_abi_version() { return 1u; }
const char* bind_device_name() { return "device"; }
}
#endif

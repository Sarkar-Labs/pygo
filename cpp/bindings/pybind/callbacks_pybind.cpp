#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_callbacks(pybind11::module_& m) {
    m.def("callbacks_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("callbacks_name", []() { return std::string("callbacks"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_callbacks_abi_version() { return 1u; }
const char* bind_callbacks_name() { return "callbacks"; }
}
#endif

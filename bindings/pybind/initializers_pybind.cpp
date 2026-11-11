#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_initializers(pybind11::module_& m) {
    m.def("initializers_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("initializers_name", []() { return std::string("initializers"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_initializers_abi_version() { return 1u; }
const char* bind_initializers_name() { return "initializers"; }
}
#endif

#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_data(pybind11::module_& m) {
    m.def("data_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("data_name", []() { return std::string("data"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_data_abi_version() { return 1u; }
const char* bind_data_name() { return "data"; }
}
#endif

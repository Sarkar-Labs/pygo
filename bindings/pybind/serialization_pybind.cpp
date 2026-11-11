#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_serialization(pybind11::module_& m) {
    m.def("serialization_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("serialization_name", []() { return std::string("serialization"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_serialization_abi_version() { return 1u; }
const char* bind_serialization_name() { return "serialization"; }
}
#endif

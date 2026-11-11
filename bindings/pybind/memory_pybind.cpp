#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_memory(pybind11::module_& m) {
    m.def("memory_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("memory_name", []() { return std::string("memory"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_memory_abi_version() { return 1u; }
const char* bind_memory_name() { return "memory"; }
}
#endif

#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_runtime(pybind11::module_& m) {
    m.def("runtime_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("runtime_name", []() { return std::string("runtime"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_runtime_abi_version() { return 1u; }
const char* bind_runtime_name() { return "runtime"; }
}
#endif

#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_distributed(pybind11::module_& m) {
    m.def("distributed_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("distributed_name", []() { return std::string("distributed"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_distributed_abi_version() { return 1u; }
const char* bind_distributed_name() { return "distributed"; }
}
#endif

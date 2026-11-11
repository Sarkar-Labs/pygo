#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_tensor(pybind11::module_& m) {
    m.def("tensor_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("tensor_name", []() { return std::string("tensor"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_tensor_abi_version() { return 1u; }
const char* bind_tensor_name() { return "tensor"; }
}
#endif

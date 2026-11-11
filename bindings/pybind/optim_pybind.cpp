#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_optim(pybind11::module_& m) {
    m.def("optim_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("optim_name", []() { return std::string("optim"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_optim_abi_version() { return 1u; }
const char* bind_optim_name() { return "optim"; }
}
#endif

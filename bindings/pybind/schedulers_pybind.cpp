#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_schedulers(pybind11::module_& m) {
    m.def("schedulers_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("schedulers_name", []() { return std::string("schedulers"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_schedulers_abi_version() { return 1u; }
const char* bind_schedulers_name() { return "schedulers"; }
}
#endif

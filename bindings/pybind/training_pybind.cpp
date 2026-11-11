#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_training(pybind11::module_& m) {
    m.def("training_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("training_name", []() { return std::string("training"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_training_abi_version() { return 1u; }
const char* bind_training_name() { return "training"; }
}
#endif

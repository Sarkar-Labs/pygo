#include <cstdint>
#include <string>

#if __has_include(<pybind11/pybind11.h>)
#include <pybind11/pybind11.h>
namespace pygo::python {
void bind_inference(pybind11::module_& m) {
    m.def("inference_abi_version", []() -> std::uint32_t { return 1u; });
    m.def("inference_name", []() { return std::string("inference"); });
}
}
#else
namespace pygo::python {
std::uint32_t bind_inference_abi_version() { return 1u; }
const char* bind_inference_name() { return "inference"; }
}
#endif

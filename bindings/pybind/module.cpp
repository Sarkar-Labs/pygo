#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include "../../cpp/bridge/c_api.h"

namespace py = pybind11;

namespace {
struct TensorView {
    void* data = nullptr;
    std::vector<std::int64_t> shape;
    std::vector<std::int64_t> strides;
    int dtype = 0;
    int device = 0;
    std::int64_t numel = 0;
    int itemsize = 0;
    void* native_handle = nullptr;
};

// The capsule is produced by the Go CGo bridge.  Keep this ABI deliberately
// small: Python borrows the storage; ownership remains with the Go tensor.
struct PyGoTensorABI {
    void* data_ptr;
    std::int64_t* shape;
    std::int64_t* strides;
    int ndim;
    int dtype;
    int device;
    std::int64_t numel;
    int itemsize;
    void* native_handle;
};

PyGoTensorABI* capsule_ptr(const py::capsule& capsule) {
    if (!capsule || std::strcmp(capsule.name(), "PyGoTensor") != 0)
        throw std::invalid_argument("expected a PyGoTensor capsule");
    auto* ptr = static_cast<PyGoTensorABI*>(capsule.get_pointer());
    if (!ptr) throw std::invalid_argument("null PyGoTensor capsule");
    return ptr;
}

TensorView view_from_capsule(const py::capsule& capsule) {
    auto* src = capsule_ptr(capsule);
    TensorView out;
    out.data = src->data_ptr;
    out.dtype = src->dtype;
    out.device = src->device;
    out.numel = src->numel;
    out.itemsize = src->itemsize;
    out.native_handle = src->native_handle;
    if (src->ndim < 0 || src->ndim > 64) throw std::invalid_argument("invalid tensor rank");
    out.shape.assign(src->shape, src->shape + src->ndim);
    out.strides.assign(src->strides, src->strides + src->ndim);
    return out;
}

std::string dtype_name(int dtype) {
    switch (dtype) {
        case 0: return "float32";
        case 1: return "float16";
        case 2: return "bfloat16";
        case 3: return "float64";
        case 4: return "int8";
        case 5: return "int64";
        default: return "unknown";
    }
}
}

PYBIND11_MODULE(_pygo_native, m) {
    m.doc() = "PyGo embedded native extension: zero-copy bridge into the C++/CUDA engine";
    m.attr("abi_version") = 1;
    m.def("runtime_name", [] { return std::string("PyGo embedded CPython runtime"); });
    m.def("architecture", [] {
        return std::string("Go -> CGo -> embedded CPython -> pybind11 -> C++/CUDA/AVX-512");
    });
    m.def("has_zero_copy", [] { return true; });

    py::class_<TensorView>(m, "TensorView")
        .def_property_readonly("data_ptr", [](const TensorView& t) {
            return reinterpret_cast<std::uintptr_t>(t.data);
        })
        .def_property_readonly("shape", [](const TensorView& t) { return t.shape; })
        .def_property_readonly("strides", [](const TensorView& t) { return t.strides; })
        .def_property_readonly("dtype", [](const TensorView& t) { return dtype_name(t.dtype); })
        .def_property_readonly("dtype_code", [](const TensorView& t) { return t.dtype; })
        .def_property_readonly("device", [](const TensorView& t) { return t.device; })
        .def_property_readonly("numel", [](const TensorView& t) { return t.numel; })
        .def_property_readonly("itemsize", [](const TensorView& t) { return t.itemsize; })
        .def_property_readonly("native_engine", [](const TensorView& t) { return t.native_handle != nullptr; });

    m.def("tensor_view", [](const py::capsule& capsule) {
        return view_from_capsule(capsule);
    }, py::arg("capsule"),
    "Create a zero-copy Python view over a Go-owned tensor capsule.");

    m.def("tensor_data_address", [](const py::capsule& capsule) {
        return reinterpret_cast<std::uintptr_t>(view_from_capsule(capsule).data);
    });

    m.def("tensor_native_sum", [](const py::capsule& capsule) {
        auto view = view_from_capsule(capsule);
        if (!view.native_handle) throw std::invalid_argument("tensor is not backed by the native C++ engine");
        auto* handle = static_cast<pygo_tensor_handle*>(view.native_handle);
        const float* data = pygo_tensor_data_f32(handle);
        if (!data) throw std::invalid_argument("native tensor is not float32");
        double sum = 0.0;
        for (std::int64_t i = 0; i < view.numel; ++i) sum += data[i];
        return sum;
    });
}

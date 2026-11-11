#pragma once
// PyGo device
//
// Canonical Device/DeviceKind definition. Every backend (CPU reference impl,
// CUDA, Metal, Vulkan, OpenCL, AVX/NEON dispatch) identifies itself through
// this type. Kept intentionally small: this header must never depend on
// tensor.h (tensor.h depends on this, not the other way around).

#include <cstdint>
#include <string>

namespace pygo {

// ─── DeviceKind ──────────────────────────────────────────────────────────────
// NOTE: values here are referenced across the codebase as `DeviceKind::CPU`,
// `DeviceKind::CUDA`, etc. (see cpp/nn/attention.cpp, cpp/optim/adamw.cpp,
// cpp/nn/batch_norm.cpp). Keep this list in sync with any new backend added
// under cpp/<backend>/.
enum class DeviceKind : uint8_t {
    CPU = 0,
    CUDA = 1,
    MPS = 2,     // Apple Metal Performance Shaders
    METAL = 3,   // raw Metal compute (distinct from MPS-accelerated ops)
    VULKAN = 4,
    OPENCL = 5,
    XLA = 6,     // TPU (and CPU/GPU-via-XLA) through Google's XLA compiler.
                 // NOTE: go/tensor/tensor.go already referenced an XLA
                 // DeviceKind before this backend existed here at all — see
                 // the ⚠️ note in cpp/bindings/cgo/tensor_cgo.h. Added so the Go
                 // and C++ enums finally agree, not because a TPU backend is
                 // implemented (it isn't — see README.md's Device section).
};

using DeviceType = DeviceKind;

std::string device_kind_name(DeviceKind kind);

// ─── Device ──────────────────────────────────────────────────────────────────
// A Device is a (kind, index) pair, e.g. {CUDA, 0} == "cuda:0".
// `index` is meaningless for CPU and is left at 0 by convention.
struct Device {
    DeviceKind kind{DeviceKind::CPU};
    int8_t     index{0};

    constexpr Device() = default;
    constexpr Device(DeviceKind k, int8_t idx = 0) : kind(k), index(idx) {}

    constexpr bool operator==(const Device& o) const noexcept {
        return kind == o.kind && index == o.index;
    }
    constexpr bool operator!=(const Device& o) const noexcept { return !(*this == o); }

    bool is_cpu()  const noexcept { return kind == DeviceKind::CPU; }
    bool is_cuda() const noexcept { return kind == DeviceKind::CUDA; }
    bool is_gpu()  const noexcept { return kind == DeviceKind::CUDA || kind == DeviceKind::MPS || kind == DeviceKind::METAL || kind == DeviceKind::VULKAN || kind == DeviceKind::OPENCL; }
    bool is_tpu()  const noexcept { return kind == DeviceKind::XLA; }

    std::string str() const;

    // Convenience default-device singletons. Defined as `static const`
    // (not constexpr) in device.cpp so `Device::CPU` can be used as a
    // default function argument the same way `Device::CUDA` is compared
    // against elsewhere in the codebase (see cpp/kernels/matmul.cpp,
    // cpp/nn/linear.cpp).
    static const Device CPU;
    static const Device kCPU;
    static const Device CUDA;
    static const Device MPS;
    static const Device METAL;
    static const Device VULKAN;
    static const Device OPENCL;
    static const Device XLA;
};

} // namespace pygo

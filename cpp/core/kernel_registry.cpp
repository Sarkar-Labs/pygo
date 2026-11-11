#include "pygo/kernel_registry.h"
#include <stdexcept>

namespace pygo::core {

std::string KernelRegistry::make_key(const std::string& op, DeviceKind dev, DType dt) {
    return op + "|" + std::to_string((int)dev) + "|" + std::to_string((int)dt);
}

KernelRegistry& KernelRegistry::instance() {
    static KernelRegistry r;
    return r;
}

void KernelRegistry::register_kernel(const std::string& op, DeviceKind dev, DType dt, KernelFn fn) {
    kernels_[make_key(op, dev, dt)] = std::move(fn);
}

bool KernelRegistry::has(const std::string& op, DeviceKind dev, DType dt) const {
    return kernels_.find(make_key(op, dev, dt)) != kernels_.end();
}

KernelFn KernelRegistry::find(const std::string& op, DeviceKind dev, DType dt) const {
    auto it = kernels_.find(make_key(op, dev, dt));
    if (it != kernels_.end()) return it->second;

    auto fallback_it = kernels_.find(make_key(op, DeviceKind::CPU, DType::Float32));
    if (fallback_it != kernels_.end()) return fallback_it->second;

    throw std::runtime_error("pygo::core::KernelRegistry: no kernel registered for op=\"" + op +
                              "\" on device=" + std::to_string((int)dev) + " dtype=" + std::to_string((int)dt) +
                              " (and no CPU/Float32 fallback either)");
}

} // namespace pygo::core

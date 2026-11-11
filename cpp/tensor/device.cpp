#include "pygo/device.h"

namespace pygo {

const Device Device::CPU    = Device(DeviceKind::CPU, 0);
const Device Device::CUDA   = Device(DeviceKind::CUDA, 0);
const Device Device::MPS    = Device(DeviceKind::MPS, 0);
const Device Device::METAL  = Device(DeviceKind::METAL, 0);
const Device Device::VULKAN = Device(DeviceKind::VULKAN, 0);
const Device Device::OPENCL = Device(DeviceKind::OPENCL, 0);
const Device Device::XLA    = Device(DeviceKind::XLA, 0);

std::string device_kind_name(DeviceKind kind) {
    switch (kind) {
        case DeviceKind::CPU:    return "cpu";
        case DeviceKind::CUDA:   return "cuda";
        case DeviceKind::MPS:    return "mps";
        case DeviceKind::METAL:  return "metal";
        case DeviceKind::VULKAN: return "vulkan";
        case DeviceKind::OPENCL: return "opencl";
        case DeviceKind::XLA:    return "xla";
    }
    return "unknown";
}

std::string Device::str() const {
    std::string s = device_kind_name(kind);
    if (kind != DeviceKind::CPU) s += ":" + std::to_string((int)index);
    return s;
}

} // namespace pygo

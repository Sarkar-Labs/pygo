#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <cstdint>

namespace pygo {

Tensor linspace(double start, double stop, int64_t steps, 
                DataType dtype = DataType::Float32, 
                const Device& device = Device(DeviceType::CPU));

} // namespace pygo
#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor arange(double start, double end, double step = 1.0, DataType dtype = DataType::Float32, Device device = Device::kCPU);

Tensor arange(double end, DataType dtype = DataType::Float32, Device device = Device::kCPU);

} // namespace pygo
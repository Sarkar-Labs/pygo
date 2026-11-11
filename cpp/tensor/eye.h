#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <cstdint>

namespace pygo {

Tensor eye(int64_t n, int64_t m = -1, 
           DataType dtype = DataType::Float32, 
           const Device& device = Device(DeviceType::CPU));

} // namespace pygo
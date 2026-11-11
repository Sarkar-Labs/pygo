#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <vector>
#include <cstdint>

namespace pygo {

Tensor empty(const std::vector<int64_t>& shape, 
             DataType dtype = DataType::Float32, 
             const Device& device = Device(DeviceType::CPU));

} // namespace pygo
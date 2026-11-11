#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <vector>
#include <cstdint>

namespace pygo {

Tensor flatten(const Tensor& input, int64_t start_dim = 0, int64_t end_dim = -1);

} // namespace pygo
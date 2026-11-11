#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <cstdint>

namespace pygo {

Tensor diff(const Tensor& input, int64_t n = 1, int64_t dim = -1);

} // namespace pygo
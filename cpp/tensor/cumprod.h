#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <cstdint>

namespace pygo {

Tensor cumprod(const Tensor& input, int64_t dim);

} // namespace pygo
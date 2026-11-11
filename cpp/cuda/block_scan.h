#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor cumsum(const Tensor& input, int64_t dim = -1);
Tensor cumprod(const Tensor& input, int64_t dim = -1);

} // namespace pygo
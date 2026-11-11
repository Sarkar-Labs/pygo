#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <cstdint>

namespace pygo {

Tensor argmin(const Tensor& input, int64_t dim, bool keepdim = false);

Tensor argmin(const Tensor& input);

} // namespace pygo
#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor cross(const Tensor& input, const Tensor& other, int64_t dim = -1);

} // namespace pygo
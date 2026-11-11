#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <cstdint>

namespace pygo {

Tensor argmax(const Tensor& input, int64_t dim, bool keepdim = false);

Tensor argmax(const Tensor& input);

} // namespace pygo
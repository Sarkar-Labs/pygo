#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor bmm(const Tensor& input, const Tensor& mat2);

} // namespace pygo
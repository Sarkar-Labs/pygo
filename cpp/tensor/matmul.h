#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor matmul(const Tensor& input1, const Tensor& input2);

} // namespace pygo
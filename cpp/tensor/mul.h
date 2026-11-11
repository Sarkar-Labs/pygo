#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor mul(const Tensor& input1, const Tensor& input2);
void mul_(Tensor& input1, const Tensor& input2);

} // namespace pygo
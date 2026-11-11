#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor cosh(const Tensor& input);

void cosh_(Tensor& input);

} // namespace pygo
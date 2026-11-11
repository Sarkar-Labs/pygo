#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor norm(const Tensor& input, double p = 2.0);

} // namespace pygo
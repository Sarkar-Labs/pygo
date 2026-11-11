#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor clip(const Tensor& input, double min_val, double max_val);

void clip_(Tensor& input, double min_val, double max_val);

} // namespace pygo
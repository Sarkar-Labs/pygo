#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor gelu(const Tensor& input);
void gelu_(Tensor& input);

} // namespace pygo
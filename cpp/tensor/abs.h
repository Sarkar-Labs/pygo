#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor abs(const Tensor& input);

void abs_(Tensor& input);

} // namespace pygo
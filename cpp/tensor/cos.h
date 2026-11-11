#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor cos(const Tensor& input);

void cos_(Tensor& input);

} // namespace pygo
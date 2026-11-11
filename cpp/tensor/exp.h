#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor exp(const Tensor& input);

void exp_(Tensor& input);

} // namespace pygo
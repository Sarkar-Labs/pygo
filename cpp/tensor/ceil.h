#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor ceil(const Tensor& input);

void ceil_(Tensor& input);

} // namespace pygo
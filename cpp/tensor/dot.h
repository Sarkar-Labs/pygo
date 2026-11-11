#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor dot(const Tensor& input, const Tensor& other);

} // namespace pygo
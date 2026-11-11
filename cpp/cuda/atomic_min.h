#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor atomic_min(Tensor& input, const Tensor& other);
void atomic_min_(Tensor& input, const Tensor& other);

} // namespace pygo
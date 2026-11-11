#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor atomic_add(Tensor& input, const Tensor& other);
void atomic_add_(Tensor& input, const Tensor& other);

} // namespace pygo
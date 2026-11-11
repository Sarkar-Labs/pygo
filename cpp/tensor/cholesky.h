#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor cholesky(const Tensor& input, bool upper = false);

} // namespace pygo
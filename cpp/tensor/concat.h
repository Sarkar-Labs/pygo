#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <vector>

namespace pygo {

Tensor concat(const std::vector<Tensor>& tensors, int64_t dim = 0);

} // namespace pygo
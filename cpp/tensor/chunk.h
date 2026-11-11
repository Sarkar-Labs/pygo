#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <vector>

namespace pygo {

std::vector<Tensor> chunk(const Tensor& input, int64_t chunks, int64_t dim = 0);

} // namespace pygo
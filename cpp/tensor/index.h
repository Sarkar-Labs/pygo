#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <cstdint>

namespace pygo {

Tensor index_select(const Tensor& input, int64_t dim, const Tensor& index);

} // namespace pygo
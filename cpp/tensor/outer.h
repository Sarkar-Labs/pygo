#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor outer(const Tensor& vec1, const Tensor& vec2);

} // namespace pygo
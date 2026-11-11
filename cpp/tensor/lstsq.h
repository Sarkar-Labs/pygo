#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor lstsq(const Tensor& A, const Tensor& B);

} // namespace pygo
#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor atomic_cas(Tensor& input, const Tensor& compare, const Tensor& val);

} // namespace pygo
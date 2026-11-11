#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <tuple>

namespace pygo {

std::tuple<Tensor, Tensor, Tensor> lu(const Tensor& input);

} // namespace pygo
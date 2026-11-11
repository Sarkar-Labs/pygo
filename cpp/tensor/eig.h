#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <utility>

namespace pygo {

std::pair<Tensor, Tensor> eig(const Tensor& input, bool eigenvectors = true);

} // namespace pygo
#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <string>
#include <vector>

namespace pygo {

Tensor einsum(const std::string& equation, const std::vector<Tensor>& operands);

} // namespace pygo
#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <vector>

namespace pygo {

Tensor reduce_sum(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);
Tensor reduce_mean(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);
Tensor reduce_max(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);
Tensor reduce_min(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);

} // namespace pygo
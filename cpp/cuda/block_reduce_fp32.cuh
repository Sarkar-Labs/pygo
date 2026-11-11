#pragma once

#include "pygo/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>
#include <vector>

namespace pygo::cuda {

Tensor reduce_sum_fp32(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);
Tensor reduce_mean_fp32(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);
Tensor reduce_max_fp32(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);
Tensor reduce_min_fp32(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);

} // namespace pygo::cuda
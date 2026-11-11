#pragma once

#include "pygo/tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>
#include <vector>

namespace pygo::cuda {

Tensor reduce_sum_fp16(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);
Tensor reduce_mean_fp16(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);
Tensor reduce_max_fp16(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);
Tensor reduce_min_fp16(const Tensor& input, const std::vector<int64_t>& dims = {}, bool keepdim = false);

} // namespace pygo::cuda
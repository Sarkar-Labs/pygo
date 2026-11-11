#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void reduce_sum_fp32(const float*, float*, int64_t);

} // namespace pygo::cuda

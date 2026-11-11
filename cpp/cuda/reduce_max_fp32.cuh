#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void reduce_max_fp32(const float*, float*, int64_t);

} // namespace pygo::cuda

#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void fused_adamw(float*, const float*, float*, float*, int64_t, float, float, float, float, float, float, int64_t);

} // namespace pygo::cuda

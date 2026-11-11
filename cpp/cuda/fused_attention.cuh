#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void fused_attention(const float*, const float*, const float*, float*, int, int, int, int, float);

} // namespace pygo::cuda

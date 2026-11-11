#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void fused_layernorm(const float*, const float*, const float*, float*, int64_t, int64_t, float);

} // namespace pygo::cuda

#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

float cross_entropy_fp32(const float* logits, const int32_t* targets, int64_t rows, int64_t classes);

} // namespace pygo::cuda

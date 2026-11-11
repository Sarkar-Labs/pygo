#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

float focal_loss_fp16(const float* logits, const int32_t* targets, int64_t rows, int64_t classes, float gamma, float alpha);

} // namespace pygo::cuda

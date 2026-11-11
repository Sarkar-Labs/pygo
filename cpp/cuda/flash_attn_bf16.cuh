#pragma once
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pygo::cuda {

void flash_attn_bf16(const void* q, const void* k, const void* v, void* out, int B, int H, int N, int D, float scale);

} // namespace pygo::cuda

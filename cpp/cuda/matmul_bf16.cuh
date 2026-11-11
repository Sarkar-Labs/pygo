#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void matmul_bf16(const float*, const float*, float*, int, int, int, float, float);

} // namespace pygo::cuda

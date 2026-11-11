#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void mma_gemm(const float*, const float*, float*, int, int, int);

} // namespace pygo::cuda

#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void topk_fp32(const float*, float*, int32_t*, int64_t, int32_t);

} // namespace pygo::cuda

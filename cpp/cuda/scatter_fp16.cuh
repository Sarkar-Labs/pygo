#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void scatter_fp16(const float*, const int64_t*, float*, int64_t, int64_t);

} // namespace pygo::cuda

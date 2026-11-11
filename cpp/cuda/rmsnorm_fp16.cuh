#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void rmsnorm_fp16(const float*, float*, int64_t, int64_t);

} // namespace pygo::cuda

#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void rope_fp32(float*, int64_t, int64_t, float);

} // namespace pygo::cuda

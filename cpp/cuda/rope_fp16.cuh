#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void rope_fp16(float*, int64_t, int64_t, float);

} // namespace pygo::cuda

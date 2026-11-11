#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void quantize_int8(const float*, uint8_t*, int64_t, float, int32_t);

} // namespace pygo::cuda

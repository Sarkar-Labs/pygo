#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void dequantize_int8(const uint8_t*, float*, int64_t, float, int32_t);

} // namespace pygo::cuda

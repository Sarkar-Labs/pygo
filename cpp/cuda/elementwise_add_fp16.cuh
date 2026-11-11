#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void elementwise_add_fp16(const __half*, const __half*, __half*, int64_t);

} // namespace pygo::cuda

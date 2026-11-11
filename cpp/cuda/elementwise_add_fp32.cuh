#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void elementwise_add_fp32(const float*, const float*, float*, int64_t);

} // namespace pygo::cuda

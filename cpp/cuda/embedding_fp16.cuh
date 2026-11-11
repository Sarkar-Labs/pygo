#pragma once
#include <cuda_fp16.h>
#include <cstdint>

namespace pygo::cuda {

void embedding_fp16(const void* weights, const int64_t* indices, void* output, int64_t count, int64_t dim);

} // namespace pygo::cuda

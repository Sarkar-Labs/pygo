#include "warp_reduce_fp16.cuh"
#include "pygo/cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <algorithm>

namespace pygo::cuda {
namespace {

__global__ void warp_reduce_fp16_kernel(const float* input, float* output, int64_t n) {
    float local = 0.0f;
    const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride)
        local += input[i];

    for (int offset = 16; offset > 0; offset >>= 1)
        local += __shfl_down_sync(0xffffffffu, local, offset);

    if ((threadIdx.x & 31) == 0)
        atomicAdd(output, local);
}

} // namespace

void warp_reduce_fp16(const float* input, float* output, int64_t n) {
    if (!input || !output || n < 0)
        throw std::invalid_argument("warp_reduce_fp16: invalid arguments");
    PYGO_CUDA_CHECK(cudaMemset(output, 0, sizeof(float)));
    if (n == 0) return;

    const int blocks = static_cast<int>(std::min<int64_t>(1024, (n + 255) / 256));
    warp_reduce_fp16_kernel<<<blocks, 256>>>(input, output, n);
    PYGO_CUDA_CHECK(cudaGetLastError());
}

} // namespace pygo::cuda

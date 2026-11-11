#include "quantize_int4.cuh"
#include "pygo/cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace pygo::cuda {
namespace {
__global__ void quantize_int4_kernel(const float* x, uint8_t* y, int64_t n, float scale, int32_t zp) {
    const int64_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i = pair * 2;
    if (i >= n) return;
    auto q = [&](float v) { return max(0, min(15, static_cast<int>(lrintf(v / scale) + zp))); };
    const uint8_t lo = static_cast<uint8_t>(q(x[i]));
    const uint8_t hi = (i + 1 < n) ? static_cast<uint8_t>(q(x[i + 1])) : 0;
    y[pair] = static_cast<uint8_t>(lo | (hi << 4));
}
}
void quantize_int4(const float* x, uint8_t* y, int64_t n, float scale, int32_t zp) {
    if (!x || !y || n < 0 || !std::isfinite(scale) || scale == 0.0f)
        throw std::invalid_argument("quantize_int4: invalid arguments");
    if (!n) return;
    const int blocks = static_cast<int>((n + 511) / 512);
    quantize_int4_kernel<<<blocks, 256>>>(x, y, n, scale, zp);
    PYGO_CUDA_CHECK(cudaGetLastError());
}
} // namespace pygo::cuda

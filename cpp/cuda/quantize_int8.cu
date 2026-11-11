#include "quantize_int8.cuh"
#include "pygo/cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace pygo::cuda {
namespace {
__global__ void quantize_int8_kernel(const float* x, uint8_t* y, int64_t n, float scale, int32_t zp) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int q = static_cast<int>(lrintf(x[i] / scale) + zp);
    q = max(0, min(255, q));
    y[i] = static_cast<uint8_t>(q);
}
}
void quantize_int8(const float* x, uint8_t* y, int64_t n, float scale, int32_t zp) {
    if (!x || !y || n < 0 || !std::isfinite(scale) || scale == 0.0f)
        throw std::invalid_argument("quantize_int8: invalid arguments");
    if (!n) return;
    quantize_int8_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(x, y, n, scale, zp);
    PYGO_CUDA_CHECK(cudaGetLastError());
}
} // namespace pygo::cuda

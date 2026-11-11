#include "dequantize_int4.cuh"
#include "pygo/cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace pygo::cuda {
namespace {
__global__ void dequantize_int4_kernel(const uint8_t* x, float* y, int64_t n, float scale, int32_t zp) {
    const int64_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i = pair * 2;
    if (i >= n) return;
    const uint8_t packed = x[pair];
    y[i] = (static_cast<int>(packed & 0x0f) - zp) * scale;
    if (i + 1 < n) y[i + 1] = (static_cast<int>((packed >> 4) & 0x0f) - zp) * scale;
}
}
void dequantize_int4(const uint8_t* x, float* y, int64_t n, float scale, int32_t zp) {
    if (!x || !y || n < 0 || scale == 0.0f) throw std::invalid_argument("dequantize_int4: invalid arguments");
    if (!n) return;
    dequantize_int4_kernel<<<static_cast<unsigned>((n + 511) / 512), 256>>>(x, y, n, scale, zp);
    PYGO_CUDA_CHECK(cudaGetLastError());
}
} // namespace pygo::cuda

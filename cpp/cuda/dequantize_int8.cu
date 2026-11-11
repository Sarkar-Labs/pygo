#include "dequantize_int8.cuh"
#include "pygo/cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace pygo::cuda {
namespace {
__global__ void dequantize_int8_kernel(const uint8_t* x, float* y, int64_t n, float scale, int32_t zp) {
    const int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (static_cast<int>(x[i]) - zp) * scale;
}
}
void dequantize_int8(const uint8_t* x, float* y, int64_t n, float scale, int32_t zp) {
    if (!x || !y || n < 0) throw std::invalid_argument("dequantize_int8: invalid arguments");
    if (!n) return;
    dequantize_int8_kernel<<<static_cast<unsigned>((n + 255) / 256), 256>>>(x, y, n, scale, zp);
    PYGO_CUDA_CHECK(cudaGetLastError());
}
} // namespace pygo::cuda

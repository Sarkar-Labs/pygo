#include "fused_mlp.cuh"
#include "pygo/cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace pygo::cuda {
namespace {

__device__ __forceinline__ float gelu(float x) {
    constexpr float k = 0.7978845608028654f;
    return 0.5f * x * (1.0f + tanhf(k * (x + 0.044715f * x * x * x)));
}

__global__ void fused_mlp_kernel(const float* x, const float* w1, const float* w2,
                                 float* y, int64_t rows, int64_t hidden, int64_t output) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const int64_t total = rows * output;
    if (index >= total) return;
    const int64_t r = index / output;
    const int64_t o = index % output;
    float acc = 0.0f;
    for (int64_t h = 0; h < hidden; ++h) {
        float first = 0.0f;
        for (int64_t k = 0; k < hidden; ++k) first += x[r * hidden + k] * w1[k * hidden + h];
        acc += gelu(first) * w2[h * output + o];
    }
    y[index] = acc;
}

} // namespace

void fused_mlp(const float* x, const float* w1, const float* w2, float* y,
               int64_t rows, int64_t hidden, int64_t output) {
    if (!x || !w1 || !w2 || !y || rows < 0 || hidden <= 0 || output <= 0)
        throw std::invalid_argument("fused_mlp: invalid arguments");
    if (!rows) return;
    const int64_t total = rows * output;
    fused_mlp_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(x, w1, w2, y, rows, hidden, output);
    PYGO_CUDA_CHECK(cudaGetLastError());
}

} // namespace pygo::cuda

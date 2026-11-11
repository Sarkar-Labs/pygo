#include "quantize_nf4.cuh"
#include "pygo/cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace pygo::cuda {
namespace {
__device__ __forceinline__ int nf4_index(float x) {
    // Standard NF4 codebook, ordered from negative to positive.
    constexpr float codebook[16] = {
        -1.000000f, -0.696193f, -0.525073f, -0.394917f,
        -0.284441f, -0.184773f, -0.091050f,  0.000000f,
         0.079580f,  0.160930f,  0.246112f,  0.337915f,
         0.440710f,  0.562617f,  0.722957f,  1.000000f
    };
    int best = 0;
    float distance = fabsf(x - codebook[0]);
    for (int i = 1; i < 16; ++i) {
        float d = fabsf(x - codebook[i]);
        if (d < distance) { distance = d; best = i; }
    }
    return best;
}
__global__ void quantize_nf4_kernel(const float* x, uint8_t* y, int64_t n, float scale, int32_t zp) {
    const int64_t pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i = pair * 2;
    if (i >= n) return;
    const float a = fminf(fmaxf(x[i] / scale, -1.0f), 1.0f);
    const float b = (i + 1 < n) ? fminf(fmaxf(x[i + 1] / scale, -1.0f), 1.0f) : 0.0f;
    y[pair] = static_cast<uint8_t>(nf4_index(a) | (nf4_index(b) << 4));
    (void)zp;
}
}
void quantize_nf4(const float* x, uint8_t* y, int64_t n, float scale, int32_t zp) {
    if (!x || !y || n < 0 || !std::isfinite(scale) || scale == 0.0f)
        throw std::invalid_argument("quantize_nf4: invalid arguments");
    if (!n) return;
    quantize_nf4_kernel<<<static_cast<unsigned>((n + 511) / 512), 256>>>(x, y, n, scale, zp);
    PYGO_CUDA_CHECK(cudaGetLastError());
}
} // namespace pygo::cuda

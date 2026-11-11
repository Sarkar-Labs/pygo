#include "tensorcore_conv.cuh"
#include "pygo/cuda.h"
#include <cuda_runtime.h>
#include <stdexcept>

namespace pygo::cuda {
namespace {

__global__ void tensorcore_conv_kernel(const float* x, const float* weight, const float* bias,
                                       float* y, int N, int C, int H, int W, int K) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * K * H * W;
    if (index >= total) return;
    const int w = index % W;
    const int h = (index / W) % H;
    const int k = (index / (H * W)) % K;
    const int n = index / (K * H * W);
    float acc = bias ? bias[k] : 0.0f;
    const int spatial = h * W + w;
    for (int c = 0; c < C; ++c) acc += x[(n * C + c) * H * W + spatial] * weight[k * C + c];
    y[index] = acc;
}

} // namespace

void tensorcore_conv(const float* x, const float* weight, const float* bias, float* y,
                     int N, int C, int H, int W, int K) {
    if (!x || !weight || !y || N < 0 || C <= 0 || H <= 0 || W <= 0 || K <= 0)
        throw std::invalid_argument("tensorcore_conv: invalid arguments");
    if (!N) return;
    const int64_t total = static_cast<int64_t>(N) * K * H * W;
    tensorcore_conv_kernel<<<static_cast<unsigned>((total + 255) / 256), 256>>>(x, weight, bias, y, N, C, H, W, K);
    PYGO_CUDA_CHECK(cudaGetLastError());
}

} // namespace pygo::cuda

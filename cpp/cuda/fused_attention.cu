#include "fused_attention.cuh"
#include "pygo/cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace pygo::cuda {
namespace {

__global__ void fused_attention_kernel(const float* q, const float* k, const float* v, float* out,
                                       int B, int H, int N, int D, float scale) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B * H * N) return;
    const int token = row % N;
    const int bh = row / N;
    const float* qv = q + (bh * N + token) * D;
    float* ov = out + (bh * N + token) * D;
    float max_score = -CUDART_INF_F;
    for (int j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += qv[d] * k[(bh * N + j) * D + d];
        max_score = fmaxf(max_score, dot * scale);
    }
    float denom = 0.0f;
    for (int j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += qv[d] * k[(bh * N + j) * D + d];
        denom += expf(dot * scale - max_score);
    }
    denom = fmaxf(denom, 1e-20f);
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            float dot = 0.0f;
            for (int x = 0; x < D; ++x) dot += qv[x] * k[(bh * N + j) * D + x];
            acc += expf(dot * scale - max_score) / denom * v[(bh * N + j) * D + d];
        }
        ov[d] = acc;
    }
}

} // namespace

void fused_attention(const float* q, const float* k, const float* v, float* out,
                     int B, int H, int N, int D, float scale) {
    if (!q || !k || !v || !out || B < 0 || H < 0 || N < 0 || D <= 0)
        throw std::invalid_argument("fused_attention: invalid arguments");
    if (!B || !H || !N) return;
    const int total = B * H * N;
    fused_attention_kernel<<<static_cast<unsigned>((total + 127) / 128), 128>>>(q, k, v, out, B, H, N, D, scale);
    PYGO_CUDA_CHECK(cudaGetLastError());
}

} // namespace pygo::cuda

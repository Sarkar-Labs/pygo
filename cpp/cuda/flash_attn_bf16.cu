#include "flash_attn_bf16.cuh"
#include "pygo/cuda.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

namespace pygo::cuda {
namespace {
__global__ void flash_attention_kernel(const __nv_bfloat16* q, const __nv_bfloat16* k,
                                       const __nv_bfloat16* v, __nv_bfloat16* out,
                                       int B, int H, int N, int D, float scale) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= B * H * N) return;
    const int token = row % N;
    const int bh = row / N;
    const __nv_bfloat16* qv = q + (bh * N + token) * D;
    __nv_bfloat16* ov = out + (bh * N + token) * D;
    float max_score = -CUDART_INF_F;
    for (int j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += __bfloat162float(qv[d]) * __bfloat162float(k[(bh * N + j) * D + d]);
        max_score = fmaxf(max_score, dot * scale);
    }
    float denom = 0.0f;
    for (int j = 0; j < N; ++j) {
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) dot += __bfloat162float(qv[d]) * __bfloat162float(k[(bh * N + j) * D + d]);
        denom += expf(dot * scale - max_score);
    }
    denom = fmaxf(denom, 1e-20f);
    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            float dot = 0.0f;
            for (int x = 0; x < D; ++x) dot += __bfloat162float(qv[x]) * __bfloat162float(k[(bh * N + j) * D + x]);
            acc += expf(dot * scale - max_score) / denom * __bfloat162float(v[(bh * N + j) * D + d]);
        }
        ov[d] = __float2bfloat16(acc);
    }
}
}

void flash_attn_bf16(const void* q, const void* k, const void* v, void* out,
                     int B, int H, int N, int D, float scale) {
    if (!q || !k || !v || !out || B < 0 || H < 0 || N < 0 || D <= 0)
        throw std::invalid_argument("flash_attn_bf16: invalid arguments");
    if (B * H * N == 0) return;
    const int threads = 128;
    flash_attention_kernel<<<(B * H * N + threads - 1) / threads, threads>>>(
        static_cast<const __nv_bfloat16*>(q), static_cast<const __nv_bfloat16*>(k),
        static_cast<const __nv_bfloat16*>(v), static_cast<__nv_bfloat16*>(out), B, H, N, D, scale);
    PYGO_CUDA_CHECK(cudaGetLastError());
}
} // namespace pygo::cuda

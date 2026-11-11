#include "pygo/cuda.h"
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace pygo::cuda {

namespace {

__global__ void attention_fp16_kernel(
    const __half* __restrict__ Q,
    const __half* __restrict__ K,
    const __half* __restrict__ V,
    __half* __restrict__ O,
    int B, int H, int N, int D, float scale) {

    int b = blockIdx.y / H;
    int h = blockIdx.y % H;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (b >= B || h >= H || i >= N) return;

    int bh_offset = (b * H + h) * N * D;
    const __half* q_ptr = Q + bh_offset + i * D;
    const __half* k_base = K + bh_offset;
    const __half* v_base = V + bh_offset;
    __half* o_ptr = O + bh_offset + i * D;

    extern __shared__ float shared_mem[];
    float* scores = shared_mem + threadIdx.x * N;

    float max_score = -1e30f;
    for (int j = 0; j < N; ++j) {
        float dot = 0.0f;
        const __half* k_ptr = k_base + j * D;
        for (int d = 0; d < D; ++d) {
            dot += __half2float(q_ptr[d]) * __half2float(k_ptr[d]);
        }
        float score = dot * scale;
        scores[j] = score;
        if (score > max_score) {
            max_score = score;
        }
    }

    float sum_exp = 0.0f;
    for (int j = 0; j < N; ++j) {
        scores[j] = expf(scores[j] - max_score);
        sum_exp += scores[j];
    }

    float inv_sum_exp = 1.0f / (sum_exp + 1e-6f);

    for (int d = 0; d < D; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < N; ++j) {
            float w = scores[j] * inv_sum_exp;
            const __half* v_ptr = v_base + j * D;
            acc += w * __half2float(v_ptr[d]);
        }
        o_ptr[d] = __float2half(acc);
    }
}

} // anonymous namespace

Tensor attention_fp16(const Tensor& q, const Tensor& k, const Tensor& v, float scale) {
    if (q.ndim() != 4 || k.ndim() != 4 || v.ndim() != 4)
        throw std::invalid_argument("pygo::cuda::attention_fp16: Q, K, V must be 4D tensors");

    int64_t B = q.shape().dims[0];
    int64_t H = q.shape().dims[1];
    int64_t N = q.shape().dims[2];
    int64_t D = q.shape().dims[3];

    if (scale < 0.0f) {
        scale = 1.0f / std::sqrt(static_cast<float>(D));
    }

    if (!is_available())
        throw std::runtime_error("pygo::cuda::attention_fp16: no CUDA device available");

    Tensor qc = q.contiguous();
    Tensor kc = k.contiguous();
    Tensor vc = v.contiguous();

    size_t bytes = (size_t)B * H * N * D * sizeof(__half);

    __half* d_q = static_cast<__half*>(to_device(qc.data<void>(), bytes));
    __half* d_k = static_cast<__half*>(to_device(kc.data<void>(), bytes));
    __half* d_v = static_cast<__half*>(to_device(vc.data<void>(), bytes));
    __half* d_o = nullptr;

    try {
        PYGO_CUDA_CHECK(cudaMalloc(&d_o, bytes));

        int threads = 256;
        dim3 grid((unsigned)((N + threads - 1) / threads), (unsigned)(B * H));
        size_t shared_mem_size = threads * N * sizeof(float);

        attention_fp16_kernel<<<grid, threads, shared_mem_size>>>(
            d_q, d_k, d_v, d_o, (int)B, (int)H, (int)N, (int)D, scale
        );

        PYGO_CUDA_CHECK(cudaGetLastError());
        PYGO_CUDA_CHECK(cudaDeviceSynchronize());

        Tensor out = empty(q.shape(), DType::Float16, Device::CPU);
        to_host(out.mutable_data<void>(), d_o, bytes);

        free_device(d_q);
        free_device(d_k);
        free_device(d_v);
        free_device(d_o);
        return out;
    } catch (...) {
        free_device(d_q);
        free_device(d_k);
        free_device(d_v);
        free_device(d_o);
        throw;
    }
}

} // namespace pygo::cuda
// Flash Attention v2 — CUDA Implementation
// Based on: Dao et al. 2022/2023 — O(N) memory, tiled SRAM compute
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include "pygo/cuda.h"
#include "pygo/tensor.h"

namespace cg = cooperative_groups;
using namespace nvcuda::wmma;

namespace pygo::cuda {

constexpr int BQ   = 64;   // tile size along Q sequence dim
constexpr int BKV  = 64;   // tile size along K/V sequence dim
constexpr int WARP = 32;

// ─── FlashAttn2 Forward Kernel ───────────────────────────────────────────────
// Q[N,H,Sq,d]  K[N,H,Sk,d]  V[N,H,Sk,d]  → O[N,H,Sq,d]
// Maintains running softmax (m, l) for numerically stable fused attention.
template<typename scalar_t, int HEAD_DIM, bool CAUSAL>
__global__ void flash_attn2_fwd_kernel(
    const scalar_t* __restrict__ Q,   // [N, H, Sq, D]
    const scalar_t* __restrict__ K,   // [N, H, Sk, D]
    const scalar_t* __restrict__ V,   // [N, H, Sk, D]
          scalar_t* __restrict__ O,   // [N, H, Sq, D]
          float*    __restrict__ L,   // [N, H, Sq]  logsumexp
    int N, int H, int Sq, int Sk,
    float scale, float alibi_slope)
{
    // Each block handles one (batch, head, tile-of-Q)
    const int batch_id = blockIdx.z;
    const int head_id  = blockIdx.y;
    const int q_tile   = blockIdx.x;    // which BQ-sized tile of Q

    const int tid   = threadIdx.x;
    const int warp  = tid / WARP;
    const int lane  = tid % WARP;

    // ── Shared memory layout ──────────────────────────────────────────────
    __shared__ scalar_t sQ  [BQ ][HEAD_DIM + 8];  // +8 pad to avoid bank conflicts
    __shared__ scalar_t sK  [BKV][HEAD_DIM + 8];
    __shared__ scalar_t sV  [BKV][HEAD_DIM + 8];
    __shared__ float    sS  [BQ ][BKV];           // attention scores tile
    __shared__ float    sm  [BQ ];                 // running max
    __shared__ float    sl  [BQ ];                 // running lse denominator

    // Pointers into global memory for this (batch, head)
    const size_t q_offset = ((size_t)batch_id * H + head_id) * Sq * HEAD_DIM;
    const size_t k_offset = ((size_t)batch_id * H + head_id) * Sk * HEAD_DIM;
    const size_t v_offset = ((size_t)batch_id * H + head_id) * Sk * HEAD_DIM;
    const size_t o_offset = q_offset;
    const size_t l_offset = ((size_t)batch_id * H + head_id) * Sq;

    // ── Load Q tile → shared ──────────────────────────────────────────────
    const int q_start = q_tile * BQ;
    #pragma unroll
    for (int r = tid; r < BQ * HEAD_DIM; r += blockDim.x) {
        int row = r / HEAD_DIM, col = r % HEAD_DIM;
        int gq  = q_start + row;
        sQ[row][col] = gq < Sq ? Q[q_offset + gq * HEAD_DIM + col]
                                : scalar_t(0);
    }

    // ── Initialise accumulators ──────────────────────────────────────────
    float acc[BQ][HEAD_DIM] = {};   // output accumulator (float32 for precision)
    float m_i[BQ];                   // running max
    float l_i[BQ];                   // running denominator
    #pragma unroll
    for (int i = tid; i < BQ; i += blockDim.x) {
        m_i[i] = -1e38f;
        l_i[i] = 0.0f;
    }
    __syncthreads();

    // ── Iterate over KV tiles ────────────────────────────────────────────
    const int num_kv_tiles = (Sk + BKV - 1) / BKV;
    for (int kv_tile = 0; kv_tile < num_kv_tiles; ++kv_tile) {
        const int kv_start = kv_tile * BKV;

        // Load K tile
        #pragma unroll
        for (int r = tid; r < BKV * HEAD_DIM; r += blockDim.x) {
            int row = r / HEAD_DIM, col = r % HEAD_DIM;
            int gk  = kv_start + row;
            sK[row][col] = gk < Sk ? K[k_offset + gk * HEAD_DIM + col]
                                    : scalar_t(0);
        }
        // Load V tile
        #pragma unroll
        for (int r = tid; r < BKV * HEAD_DIM; r += blockDim.x) {
            int row = r / HEAD_DIM, col = r % HEAD_DIM;
            int gk  = kv_start + row;
            sV[row][col] = gk < Sk ? V[v_offset + gk * HEAD_DIM + col]
                                    : scalar_t(0);
        }
        __syncthreads();

        // ── Compute S = Q @ K^T * scale ──────────────────────────────────
        for (int qi = tid; qi < BQ; qi += blockDim.x) {
            for (int ki = 0; ki < BKV; ++ki) {
                float s = 0.0f;
                #pragma unroll
                for (int d = 0; d < HEAD_DIM; ++d)
                    s += float(sQ[qi][d]) * float(sK[ki][d]);
                s *= scale;

                // Optional: ALiBi bias
                if (alibi_slope != 0.0f) {
                    int gqi = q_start + qi, gki = kv_start + ki;
                    s += alibi_slope * (gki - gqi);
                }

                // Causal mask
                if constexpr (CAUSAL) {
                    int gqi = q_start + qi, gki = kv_start + ki;
                    if (gki > gqi) s = -1e38f;
                }

                sS[qi][ki] = s;
            }
        }
        __syncthreads();

        // ── Online softmax update (Flash Attention trick) ─────────────────
        for (int qi = tid; qi < BQ; qi += blockDim.x) {
            float m_new = m_i[qi];
            for (int ki = 0; ki < BKV; ++ki)
                m_new = fmaxf(m_new, sS[qi][ki]);

            float l_new = 0.0f;
            for (int ki = 0; ki < BKV; ++ki) {
                sS[qi][ki] = expf(sS[qi][ki] - m_new);
                l_new += sS[qi][ki];
            }

            float alpha = expf(m_i[qi] - m_new);
            l_i[qi]     = alpha * l_i[qi] + l_new;

            // Rescale old accumulator
            for (int d = 0; d < HEAD_DIM; ++d)
                acc[qi][d] *= alpha;

            // Accumulate new P @ V
            for (int ki = 0; ki < BKV; ++ki)
                for (int d = 0; d < HEAD_DIM; ++d)
                    acc[qi][d] += sS[qi][ki] * float(sV[ki][d]);

            m_i[qi] = m_new;
        }
        __syncthreads();
    }

    // ── Write output & logsumexp ─────────────────────────────────────────
    for (int qi = tid; qi < BQ; qi += blockDim.x) {
        int gq = q_start + qi;
        if (gq >= Sq) continue;
        float inv_l = 1.0f / l_i[qi];
        for (int d = 0; d < HEAD_DIM; ++d)
            O[o_offset + gq * HEAD_DIM + d] = scalar_t(acc[qi][d] * inv_l);
        // Save log-sum-exp for backward
        L[l_offset + gq] = m_i[qi] + logf(l_i[qi]);
    }
}

// ─── Dispatcher ──────────────────────────────────────────────────────────────
Tensor flash_attention_v2(
    const Tensor& Q,   // [B, H, Sq, D]
    const Tensor& K,   // [B, H, Sk, D]
    const Tensor& V,   // [B, H, Sk, D]
    bool  causal        = false,
    float softmax_scale = -1.0f,
    float alibi_slope   = 0.0f)
{
    const int B  = Q.dim(0), H  = Q.dim(1);
    const int Sq = Q.dim(2), Sk = K.dim(2);
    const int D  = Q.dim(3);

    if (softmax_scale < 0) softmax_scale = 1.0f / sqrtf((float)D);

    auto O = empty({B, H, Sq, D}, Q.dtype(), Q.device());
    auto L = empty({B, H, Sq},    DType::Float32, Q.device());

    dim3 grid((Sq + BQ - 1) / BQ, H, B);
    dim3 block(256);

    auto launch = [&]<typename T, int HDIM, bool CAU>() {
        flash_attn2_fwd_kernel<T, HDIM, CAU><<<grid, block>>>(
            Q.data<T>(), K.data<T>(), V.data<T>(),
            O.data<T>(), L.data<float>(),
            B, H, Sq, Sk, softmax_scale, alibi_slope);
    };

    // Dispatch on dtype × head_dim × causal
    if (Q.dtype() == DType::Float16) {
        if (D == 64)  { causal ? launch.template operator()<__half,64,true>()
                               : launch.template operator()<__half,64,false>(); }
        else if (D == 128) { causal ? launch.template operator()<__half,128,true>()
                                    : launch.template operator()<__half,128,false>(); }
    } else if (Q.dtype() == DType::BFloat16) {
        if (D == 128) { causal ? launch.template operator()<__nv_bfloat16,128,true>()
                               : launch.template operator()<__nv_bfloat16,128,false>(); }
    } else {
        // fp32 fallback
        if (D == 64)  { causal ? launch.template operator()<float,64,true>()
                               : launch.template operator()<float,64,false>(); }
        else           { causal ? launch.template operator()<float,128,true>()
                               : launch.template operator()<float,128,false>(); }
    }

    CUDA_CHECK(cudaGetLastError());
    return O;
}

} // namespace pygo::cuda

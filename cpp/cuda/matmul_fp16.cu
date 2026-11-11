// CUTLASS-inspired FP16 GEMM using Tensor Cores (WMMA API)
// C = alpha * A @ B + beta * C
// A: [M, K]  B: [K, N]  C: [M, N]  all fp16
#include <cuda.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cooperative_groups.h>
#include <stdexcept>
#include "pygo/cuda.h"

namespace cg = cooperative_groups;
using namespace nvcuda::wmma;

namespace pygo::cuda {

// ─── WMMA tile sizes ─────────────────────────────────────────────────────────
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
// Thread-block tile
constexpr int BM  = 128;   // rows of A handled per block
constexpr int BN  = 128;   // cols of B handled per block
constexpr int BK  = 32;    // inner dim tile
constexpr int WM  = 64;    // warp tile row
constexpr int WN  = 64;    // warp tile col
constexpr int WARPS_PER_BLOCK = (BM / WM) * (BN / WN);  // = 4

// ─── Shared memory layout ─────────────────────────────────────────────────────
// Pad by 8 half-words to avoid shared-memory bank conflicts
constexpr int SMEM_A_PAD = 8;
constexpr int SMEM_B_PAD = 8;

__global__
__launch_bounds__(WARPS_PER_BLOCK * 32, 2)
void hgemm_wmma_kernel(
    const __half* __restrict__ A,   // [M, K]
    const __half* __restrict__ B,   // [K, N]
          __half* __restrict__ C,   // [M, N]
    int M, int N, int K,
    float alpha, float beta)
{
    __shared__ __half sA[BM][BK + SMEM_A_PAD];
    __shared__ __half sB[BK][BN + SMEM_B_PAD];

    const int warp_id = threadIdx.x / 32;
    const int lane    = threadIdx.x % 32;
    const int warp_row = warp_id / (BN / WN);
    const int warp_col = warp_id % (BN / WN);

    // Global tile origin for this block
    const int block_row = blockIdx.y * BM;
    const int block_col = blockIdx.x * BN;

    // WMMA fragments — one per warp tile
    constexpr int WFRAG_M = WM / WMMA_M;  // = 4
    constexpr int WFRAG_N = WN / WMMA_N;  // = 4
    constexpr int WFRAG_K = BK / WMMA_K;  // = 2

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[WFRAG_M][WFRAG_N];
    #pragma unroll
    for (int i = 0; i < WFRAG_M; ++i)
        for (int j = 0; j < WFRAG_N; ++j)
            fill_fragment(acc[i][j], 0.f);

    fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, row_major> fa[WFRAG_M][WFRAG_K];
    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, row_major> fb[WFRAG_K][WFRAG_N];

    // ── Main K loop ──────────────────────────────────────────────────────────
    for (int k_tile = 0; k_tile < (K + BK - 1) / BK; ++k_tile) {
        const int k_off = k_tile * BK;

        // ── Load A tile → shared  (coalesced, vectorised) ─────────────────
        for (int r = threadIdx.x; r < BM * BK / 8; r += blockDim.x) {
            int row = r * 8 / BK, col = (r * 8) % BK;
            int ga_row = block_row + row, ga_col = k_off + col;
            if (ga_row < M && ga_col + 7 < K) {
                // 128-bit load (8 fp16 = 16 bytes)
                *reinterpret_cast<float4*>(&sA[row][col]) =
                    *reinterpret_cast<const float4*>(&A[ga_row * K + ga_col]);
            } else {
                // Boundary guard
                for (int x = 0; x < 8; ++x)
                    sA[row][col + x] = (ga_row < M && ga_col + x < K)
                        ? A[ga_row * K + ga_col + x] : __half(0);
            }
        }

        // ── Load B tile → shared ──────────────────────────────────────────
        for (int r = threadIdx.x; r < BK * BN / 8; r += blockDim.x) {
            int row = r * 8 / BN, col = (r * 8) % BN;
            int gb_row = k_off + row, gb_col = block_col + col;
            if (gb_row < K && gb_col + 7 < N) {
                *reinterpret_cast<float4*>(&sB[row][col]) =
                    *reinterpret_cast<const float4*>(&B[gb_row * N + gb_col]);
            } else {
                for (int x = 0; x < 8; ++x)
                    sB[row][col + x] = (gb_row < K && gb_col + x < N)
                        ? B[gb_row * N + gb_col + x] : __half(0);
            }
        }

        __syncthreads();

        // ── WMMA MMA: fa × fb → acc ──────────────────────────────────────
        #pragma unroll
        for (int ki = 0; ki < WFRAG_K; ++ki) {
            #pragma unroll
            for (int mi = 0; mi < WFRAG_M; ++mi) {
                int sA_row = warp_row * WM + mi * WMMA_M;
                int sA_col = ki * WMMA_K;
                load_matrix_sync(fa[mi][ki],
                    &sA[sA_row][sA_col], BK + SMEM_A_PAD);
            }
            #pragma unroll
            for (int ni = 0; ni < WFRAG_N; ++ni) {
                int sB_row = ki * WMMA_K;
                int sB_col = warp_col * WN + ni * WMMA_N;
                load_matrix_sync(fb[ki][ni],
                    &sB[sB_row][sB_col], BN + SMEM_B_PAD);
            }
            #pragma unroll
            for (int mi = 0; mi < WFRAG_M; ++mi)
                #pragma unroll
                for (int ni = 0; ni < WFRAG_N; ++ni)
                    mma_sync(acc[mi][ni], fa[mi][ki], fb[ki][ni], acc[mi][ni]);
        }

        __syncthreads();
    }

    // ── Write output with alpha/beta scaling ─────────────────────────────────
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, __half> out_frag;

    #pragma unroll
    for (int mi = 0; mi < WFRAG_M; ++mi) {
        #pragma unroll
        for (int ni = 0; ni < WFRAG_N; ++ni) {
            int c_row = block_row + warp_row * WM + mi * WMMA_M;
            int c_col = block_col + warp_col * WN + ni * WMMA_N;
            if (c_row >= M || c_col >= N) continue;

            // Scale accumulator by alpha
            #pragma unroll
            for (int e = 0; e < acc[mi][ni].num_elements; ++e)
                acc[mi][ni].x[e] *= alpha;

            // Optionally add beta * C
            if (beta != 0.f) {
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, __half> c_frag;
                load_matrix_sync(c_frag, &C[c_row * N + c_col], N, mem_row_major);
                #pragma unroll
                for (int e = 0; e < c_frag.num_elements; ++e)
                    acc[mi][ni].x[e] += beta * float(c_frag.x[e]);
            }

            // Convert fp32 acc → fp16 and store
            #pragma unroll
            for (int e = 0; e < out_frag.num_elements; ++e)
                out_frag.x[e] = __float2half(acc[mi][ni].x[e]);
            store_matrix_sync(&C[c_row * N + c_col], out_frag, N, mem_row_major);
        }
    }
}

// ─── Batched GEMM ────────────────────────────────────────────────────────────

__global__ void batched_hgemm_kernel(
    const __half** A_ptrs,
    const __half** B_ptrs,
          __half** C_ptrs,
    int M, int N, int K, float alpha, float beta, int batch)
{
    // Each block handles one (batch_idx, tile_row, tile_col)
    int b = blockIdx.z;
    if (b >= batch) return;
    // Reuse the 2D kernel logic with batch-indexed pointers
    // (inline for simplicity; production uses device function)
    hgemm_wmma_kernel<<<dim3((N+BN-1)/BN, (M+BM-1)/BM), WARPS_PER_BLOCK*32>>>(
        A_ptrs[b], B_ptrs[b], C_ptrs[b], M, N, K, alpha, beta);
}

// ─── Host launcher ───────────────────────────────────────────────────────────

void hgemm(
    const void* A, const void* B, void* C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(WARPS_PER_BLOCK * 32);

    hgemm_wmma_kernel<<<grid, block, 0, stream>>>(
        reinterpret_cast<const __half*>(A),
        reinterpret_cast<const __half*>(B),
        reinterpret_cast<__half*>(C),
        M, N, K, alpha, beta);

    CUDA_CHECK(cudaGetLastError());
}

} // namespace pygo::cuda

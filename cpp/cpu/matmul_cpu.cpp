// High-performance CPU GEMM with AVX2 / AVX-512 / NEON intrinsics
// C = alpha * A @ B + beta * C
#include "pygo/cpu.h"
#include "pygo/simd.h"
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <omp.h>

#if defined(__AVX512F__)
#  include <immintrin.h>
#  define HAVE_AVX512 1
#elif defined(__AVX2__)
#  include <immintrin.h>
#  define HAVE_AVX2 1
#elif defined(__ARM_NEON)
#  include <arm_neon.h>
#  define HAVE_NEON 1
#endif

namespace pygo::cpu {

// ─── Cache size heuristics ────────────────────────────────────────────────────
static constexpr int L1_CACHE   = 32   * 1024;   // 32 KB
static constexpr int L2_CACHE   = 256  * 1024;   // 256 KB
static constexpr int L3_CACHE   = 8192 * 1024;   // 8 MB

// Block sizes derived from cache analysis
static constexpr int MC  = 96;    // rows of A per block (fits A into L1)
static constexpr int KC  = 256;   // inner dim per block (fits B panel into L2)
static constexpr int NC  = 3072;  // cols of B per block (fits C into L3)
static constexpr int MR  = 8;     // micro-kernel rows (register tile, 8 × AVX float)
static constexpr int NR  = 16;    // micro-kernel cols

// ─── AVX2 Micro-kernel 8×16 ──────────────────────────────────────────────────
#if defined(HAVE_AVX2)

// Compute 8×16 output tile C[MR,NR] += A[MR,KC] @ B[KC,NR]
// Uses 2×8 = 16 YMM registers for C, + 2 for B, + 1 for A broadcast
static inline void micro_kernel_8x16_avx2(
    const float* __restrict__ A,   // packed column-major [KC, MR]
    const float* __restrict__ B,   // packed row-major    [KC, NR]
    float*       __restrict__ C,   // row-major           [MR, ldc]
    int KC, int ldc)
{
    __m256 c00, c01, c10, c11, c20, c21, c30, c31,
           c40, c41, c50, c51, c60, c61, c70, c71;
    c00=c01=c10=c11=c20=c21=c30=c31=
    c40=c41=c50=c51=c60=c61=c70=c71=_mm256_setzero_ps();

    for (int k = 0; k < KC; ++k) {
        // Load one column of B (16 floats = 2 × 256-bit)
        __m256 b0 = _mm256_load_ps(B + k * NR);
        __m256 b1 = _mm256_load_ps(B + k * NR + 8);

        // Broadcast A values and FMA
        __m256 a;
        #define FMA_ROW(row) \
            a = _mm256_broadcast_ss(A + k * MR + row); \
            c##row##0 = _mm256_fmadd_ps(a, b0, c##row##0); \
            c##row##1 = _mm256_fmadd_ps(a, b1, c##row##1);

        FMA_ROW(0) FMA_ROW(1) FMA_ROW(2) FMA_ROW(3)
        FMA_ROW(4) FMA_ROW(5) FMA_ROW(6) FMA_ROW(7)
        #undef FMA_ROW
    }

    // Write back to C (add to existing values)
    #define STORE_ROW(row) \
        _mm256_storeu_ps(C + row * ldc,     \
            _mm256_add_ps(_mm256_loadu_ps(C + row * ldc),     c##row##0)); \
        _mm256_storeu_ps(C + row * ldc + 8, \
            _mm256_add_ps(_mm256_loadu_ps(C + row * ldc + 8), c##row##1));

    STORE_ROW(0) STORE_ROW(1) STORE_ROW(2) STORE_ROW(3)
    STORE_ROW(4) STORE_ROW(5) STORE_ROW(6) STORE_ROW(7)
    #undef STORE_ROW
}

#endif // HAVE_AVX2

#if defined(HAVE_AVX512)

// AVX-512 micro-kernel 8×32 (uses ZMM registers for 512-bit = 16 floats)
static inline void micro_kernel_8x32_avx512(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int KC, int ldc)
{
    __m512 c0, c1, c2, c3, c4, c5, c6, c7;
    __m512 c0b,c1b,c2b,c3b,c4b,c5b,c6b,c7b;
    c0=c1=c2=c3=c4=c5=c6=c7=_mm512_setzero_ps();
    c0b=c1b=c2b=c3b=c4b=c5b=c6b=c7b=_mm512_setzero_ps();

    for (int k = 0; k < KC; ++k) {
        __m512 b0 = _mm512_loadu_ps(B + k * 32);
        __m512 b1 = _mm512_loadu_ps(B + k * 32 + 16);
        #define FMA512(row) \
            { __m512 a = _mm512_set1_ps(A[k * 8 + row]); \
              c##row  = _mm512_fmadd_ps(a, b0, c##row);   \
              c##row##b= _mm512_fmadd_ps(a, b1, c##row##b); }
        FMA512(0) FMA512(1) FMA512(2) FMA512(3)
        FMA512(4) FMA512(5) FMA512(6) FMA512(7)
        #undef FMA512
    }

    #define STORE512(row) \
        _mm512_storeu_ps(C + row*ldc,    _mm512_add_ps(_mm512_loadu_ps(C+row*ldc),    c##row)); \
        _mm512_storeu_ps(C + row*ldc+16, _mm512_add_ps(_mm512_loadu_ps(C+row*ldc+16), c##row##b));
    STORE512(0) STORE512(1) STORE512(2) STORE512(3)
    STORE512(4) STORE512(5) STORE512(6) STORE512(7)
    #undef STORE512
}

#endif // HAVE_AVX512

// ─── Packing routines ─────────────────────────────────────────────────────────
// Pack A panel [mc, kc] into column-major block of MR columns
static void pack_A(const float* A, float* packed,
                   int mc, int kc, int lda) {
    for (int i = 0; i < mc; i += MR) {
        int ii = std::min(MR, mc - i);
        for (int k = 0; k < kc; ++k) {
            for (int r = 0; r < MR; ++r) {
                packed[k * MR + r] = (r < ii) ? A[(i + r) * lda + k] : 0.f;
            }
        }
        packed += kc * MR;
    }
}

// Pack B panel [kc, nc] into row-major block of NR rows
static void pack_B(const float* B, float* packed,
                   int kc, int nc, int ldb) {
    for (int j = 0; j < nc; j += NR) {
        int jj = std::min(NR, nc - j);
        for (int k = 0; k < kc; ++k) {
            for (int c = 0; c < NR; ++c) {
                packed[k * NR + c] = (c < jj) ? B[k * ldb + j + c] : 0.f;
            }
        }
        packed += kc * NR;
    }
}

// ─── GEMM (single-threaded inner, OpenMP outer) ───────────────────────────────
void sgemm(
    int M, int N, int K,
    float alpha,
    const float* A, int lda,
    const float* B, int ldb,
    float  beta,
    float* C, int ldc)
{
    // Scale C by beta
    if (beta == 0.f) {
        for (int i = 0; i < M; ++i)
            memset(C + i * ldc, 0, N * sizeof(float));
    } else if (beta != 1.f) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i * ldc + j] *= beta;
    }

    // Allocate packing buffers
    alignas(64) thread_local float packedA[MC * KC];
    alignas(64) thread_local float packedB[KC * NC];

    #pragma omp parallel for schedule(dynamic) collapse(2)
    for (int j = 0; j < N; j += NC) {
        for (int k = 0; k < K; k += KC) {
            int nc = std::min(NC, N - j);
            int kc = std::min(KC, K - k);

            pack_B(&B[k * ldb + j], packedB, kc, nc, ldb);

            for (int i = 0; i < M; i += MC) {
                int mc = std::min(MC, M - i);
                pack_A(&A[i * lda + k], packedA, mc, kc, lda);

                // Micro-kernel dispatch
                const float* pa = packedA;
                for (int ii = 0; ii < mc; ii += MR, pa += MR * kc) {
                    const float* pb = packedB;
                    for (int jj = 0; jj < nc; jj += NR, pb += NR * kc) {
                        float* cc = &C[(i + ii) * ldc + j + jj];
#if defined(HAVE_AVX512)
                        if (mc - ii >= MR && nc - jj >= 32)
                            micro_kernel_8x32_avx512(pa, pb, cc, kc, ldc);
                        else
#elif defined(HAVE_AVX2)
                        if (mc - ii >= MR && nc - jj >= NR)
                            micro_kernel_8x16_avx2(pa, pb, cc, kc, ldc);
                        else
#endif
                        {
                            // Scalar fallback
                            int mr2 = std::min(MR, mc - ii);
                            int nr2 = std::min(NR, nc - jj);
                            for (int r = 0; r < mr2; ++r)
                                for (int s = 0; s < nr2; ++s)
                                    for (int t = 0; t < kc; ++t)
                                        cc[r * ldc + s] += pa[t * MR + r] * pb[t * NR + s];
                        }
                    }
                }
            }
        }
    }

    // Apply alpha scaling
    if (alpha != 1.f) {
        for (int i = 0; i < M; ++i)
            for (int j = 0; j < N; ++j)
                C[i * ldc + j] *= alpha;
    }
}

// ─── FP16 → FP32 dequant GEMM ────────────────────────────────────────────────
void hgemm_cpu(
    int M, int N, int K,
    float alpha,
    const uint16_t* A, int lda,
    const uint16_t* B, int ldb,
    float beta,
    uint16_t* C, int ldc)
{
    // Dequant → fp32 → sgemm → quant
    auto a32 = std::make_unique<float[]>(M * K);
    auto b32 = std::make_unique<float[]>(K * N);
    auto c32 = std::make_unique<float[]>(M * N);

    for (int i = 0; i < M * K; ++i) a32[i] = fp16_to_fp32(A[i]);
    for (int i = 0; i < K * N; ++i) b32[i] = fp16_to_fp32(B[i]);
    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            c32[i*N+j] = fp16_to_fp32(C[i*ldc+j]);

    sgemm(M, N, K, alpha, a32.get(), K, b32.get(), N, beta, c32.get(), N);

    for (int i = 0; i < M; ++i)
        for (int j = 0; j < N; ++j)
            C[i*ldc+j] = fp32_to_fp16(c32[i*N+j]);
}

// ─── Batched GEMM ─────────────────────────────────────────────────────────────
void batched_sgemm(
    int batch, int M, int N, int K,
    float alpha,
    const float* const* A, int lda,
    const float* const* B, int ldb,
    float beta,
    float** C, int ldc)
{
    #pragma omp parallel for
    for (int b = 0; b < batch; ++b)
        sgemm(M, N, K, alpha, A[b], lda, B[b], ldb, beta, C[b], ldc);
}

} // namespace pygo::cpu

// Fused elementwise CUDA kernels — handles arbitrary op chains in one pass
// to maximize arithmetic intensity and minimize memory traffic.
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cmath>
#include "pygo/cuda.h"

namespace pygo::kernels {

// ─── Op codes for fused kernel dispatch ──────────────────────────────────────
enum class ElemOp : uint8_t {
    Add=0, Sub=1, Mul=2, Div=3, Pow=4,
    Relu=5, Gelu=6, Silu=7, Tanh=8, Sigmoid=9,
    Log=10, Exp=11, Sqrt=12, Abs=13, Neg=14,
    AddScalar=15, MulScalar=16, ClampMin=17, ClampMax=18,
    LeakyRelu=19, Elu=20, Selu=21, Mish=22, HardSwish=23,
};

// ─── Activation functions (device code) ──────────────────────────────────────

__device__ __forceinline__ float gelu_f(float x) {
    // Fast tanh approximation: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    constexpr float k0 = 0.7978845608f;  // sqrt(2/pi)
    constexpr float k1 = 0.044715f;
    float cube = x * x * x;
    return 0.5f * x * (1.0f + tanhf(k0 * (x + k1 * cube)));
}

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.0f + expf(-x));
}

__device__ __forceinline__ float mish_f(float x) {
    return x * tanhf(log1pf(expf(x)));
}

__device__ __forceinline__ float hardswish_f(float x) {
    float h3 = x + 3.0f;
    return x * fminf(fmaxf(h3, 0.0f), 6.0f) / 6.0f;
}

__device__ __forceinline__ float elu_f(float x, float alpha = 1.0f) {
    return x >= 0.0f ? x : alpha * (expf(x) - 1.0f);
}

__device__ __forceinline__ float selu_f(float x) {
    constexpr float alpha = 1.6732632423543772f;
    constexpr float scale = 1.0507009873554805f;
    return scale * (x >= 0.0f ? x : alpha * (expf(x) - 1.0f));
}

__device__ __forceinline__ float leaky_relu_f(float x, float neg_slope = 0.01f) {
    return x >= 0.0f ? x : neg_slope * x;
}

// ─── Single-element dispatch ──────────────────────────────────────────────────

__device__ __forceinline__
float apply_op(float a, float b, ElemOp op, float scalar) {
    switch (op) {
        case ElemOp::Add:       return a + b;
        case ElemOp::Sub:       return a - b;
        case ElemOp::Mul:       return a * b;
        case ElemOp::Div:       return a / b;
        case ElemOp::Pow:       return powf(a, b);
        case ElemOp::Relu:      return fmaxf(a, 0.0f);
        case ElemOp::Gelu:      return gelu_f(a);
        case ElemOp::Silu:      return silu_f(a);
        case ElemOp::Tanh:      return tanhf(a);
        case ElemOp::Sigmoid:   return 1.0f / (1.0f + expf(-a));
        case ElemOp::Log:       return logf(a);
        case ElemOp::Exp:       return expf(a);
        case ElemOp::Sqrt:      return sqrtf(a);
        case ElemOp::Abs:       return fabsf(a);
        case ElemOp::Neg:       return -a;
        case ElemOp::AddScalar: return a + scalar;
        case ElemOp::MulScalar: return a * scalar;
        case ElemOp::ClampMin:  return fmaxf(a, scalar);
        case ElemOp::ClampMax:  return fminf(a, scalar);
        case ElemOp::LeakyRelu: return leaky_relu_f(a, scalar);
        case ElemOp::Elu:       return elu_f(a, scalar);
        case ElemOp::Selu:      return selu_f(a);
        case ElemOp::Mish:      return mish_f(a);
        case ElemOp::HardSwish: return hardswish_f(a);
        default:                return a;
    }
}

// ─── Vectorised FP32 elementwise kernel (float4 = 128-bit loads) ─────────────

template<ElemOp Op, bool HasB>
__global__ void elementwise_fp32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
          float* __restrict__ C,
    int64_t n, float scalar)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;
    int64_t n4 = n / 4;

    // Vectorised path (4 floats = 128-bit load/store per thread)
    const float4* A4 = reinterpret_cast<const float4*>(A);
    const float4* B4 = reinterpret_cast<const float4*>(B);
          float4* C4 = reinterpret_cast<float4*>(C);

    for (int64_t i = tid; i < n4; i += stride) {
        float4 a = A4[i];
        float4 b = HasB ? B4[i] : float4{0, 0, 0, 0};
        float4 c;
        c.x = apply_op(a.x, b.x, Op, scalar);
        c.y = apply_op(a.y, b.y, Op, scalar);
        c.z = apply_op(a.z, b.z, Op, scalar);
        c.w = apply_op(a.w, b.w, Op, scalar);
        C4[i] = c;
    }

    // Scalar tail
    for (int64_t i = n4 * 4 + tid; i < n; i += stride) {
        float b_val = HasB ? B[i] : 0.0f;
        C[i] = apply_op(A[i], b_val, Op, scalar);
    }
}

// ─── FP16 elementwise kernel ─────────────────────────────────────────────────

template<ElemOp Op, bool HasB>
__global__ void elementwise_fp16_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
          __half* __restrict__ C,
    int64_t n, float scalar)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    // Use half2 for paired SIMD instructions on SM 7.0+
    const half2* A2 = reinterpret_cast<const half2*>(A);
    const half2* B2 = reinterpret_cast<const half2*>(B);
          half2* C2 = reinterpret_cast<half2*>(C);
    int64_t n2 = n / 2;

    for (int64_t i = tid; i < n2; i += stride) {
        half2 a = A2[i];
        half2 b = HasB ? B2[i] : half2{__half(0), __half(0)};
        float ax = __half2float(a.x), ay = __half2float(a.y);
        float bx = __half2float(b.x), by = __half2float(b.y);
        C2[i] = {__float2half(apply_op(ax, bx, Op, scalar)),
                 __float2half(apply_op(ay, by, Op, scalar))};
    }

    // Tail
    for (int64_t i = n2*2 + tid; i < n; i += stride) {
        float bv = HasB ? __half2float(B[i]) : 0.0f;
        C[i] = __float2half(apply_op(__half2float(A[i]), bv, Op, scalar));
    }
}

// ─── BF16 elementwise kernel ─────────────────────────────────────────────────

template<ElemOp Op, bool HasB>
__global__ void elementwise_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
          __nv_bfloat16* __restrict__ C,
    int64_t n, float scalar)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float av = __bfloat162float(A[i]);
        float bv = HasB ? __bfloat162float(B[i]) : 0.0f;
        C[i] = __float2bfloat16(apply_op(av, bv, Op, scalar));
    }
}

// ─── Fused chain kernel: apply N ops sequentially in a single kernel ──────────
// e.g. Linear → GELU → Dropout all fused into one memory pass

constexpr int MAX_FUSED_OPS = 8;

struct FusedOpChain {
    ElemOp ops[MAX_FUSED_OPS];
    float  scalars[MAX_FUSED_OPS];
    int    num_ops;
};

__global__ void fused_elementwise_fp32(
    const float* __restrict__ A,
          float* __restrict__ C,
    int64_t n,
    FusedOpChain chain)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    int64_t stride = blockDim.x * gridDim.x;

    for (int64_t i = tid; i < n; i += stride) {
        float x = A[i];
        for (int j = 0; j < chain.num_ops; ++j) {
            x = apply_op(x, 0.0f, chain.ops[j], chain.scalars[j]);
        }
        C[i] = x;
    }
}

// ─── Dropout (Bernoulli mask + scale) ────────────────────────────────────────

__global__ void dropout_fp32_kernel(
    const float* __restrict__ A,
          float* __restrict__ C,
          uint8_t* __restrict__ mask,
    int64_t n, float p, uint64_t seed, uint64_t offset)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    float scale = 1.0f / (1.0f - p);

    for (int64_t i = tid; i < n; i += blockDim.x * gridDim.x) {
        // Philox-4x32-10 PRNG inline (simplified 1-round version)
        uint64_t state = (seed ^ (uint64_t)i) * 6364136223846793005ULL + offset;
        float u = (float)(state >> 11) * (1.0f / float(1ULL << 53));
        bool keep = u > p;
        mask[i] = keep ? 1 : 0;
        C[i] = keep ? A[i] * scale : 0.0f;
    }
}

// ─── Layer Scale (DeiT / CaiT) ───────────────────────────────────────────────

__global__ void layer_scale_fp32(
    float* __restrict__ x, const float* __restrict__ gamma,
    int64_t B, int64_t S, int64_t D)
{
    int64_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= B * S * D) return;
    int64_t d = tid % D;
    x[tid] *= gamma[d];
}

// ─── RMS Norm forward ─────────────────────────────────────────────────────────

__global__ void rms_norm_fp32(
    const float* __restrict__ x,
    const float* __restrict__ weight,
          float* __restrict__ out,
    int64_t B, int64_t S, int64_t D, float eps)
{
    // One warp per row (B*S rows total)
    int64_t row = blockIdx.x;
    if (row >= B * S) return;

    const float* row_x = x + row * D;
    float*       row_o = out + row * D;

    // Compute RMS via warp reduce
    float rms_sq = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        rms_sq += row_x[d] * row_x[d];
    }
    // Warp reduce sum
    for (int offset = 16; offset > 0; offset >>= 1)
        rms_sq += __shfl_down_sync(0xffffffff, rms_sq, offset);
    rms_sq = __shfl_sync(0xffffffff, rms_sq, 0) / float(D) + eps;
    float inv_rms = rsqrtf(rms_sq);

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        row_o[d] = row_x[d] * inv_rms * weight[d];
    }
}

// ─── Host launcher ────────────────────────────────────────────────────────────

void launch_elementwise(
    void* A, void* B, void* C,
    int64_t n, ElemOp op, float scalar,
    bool has_b, DType dtype, cudaStream_t stream)
{
    int threads = 256;
    int64_t blocks = (n / 4 + threads - 1) / threads;
    blocks = std::min(blocks, (int64_t)65535);

    #define LAUNCH(OP, HAS_B) \
        if (dtype == DType::Float32) \
            elementwise_fp32_kernel<OP,HAS_B><<<blocks,threads,0,stream>>>( \
                (float*)A,(float*)B,(float*)C,n,scalar); \
        else if (dtype == DType::Float16) \
            elementwise_fp16_kernel<OP,HAS_B><<<blocks,threads,0,stream>>>( \
                (__half*)A,(__half*)B,(__half*)C,n,scalar); \
        else if (dtype == DType::BFloat16) \
            elementwise_bf16_kernel<OP,HAS_B><<<blocks,threads,0,stream>>>( \
                (__nv_bfloat16*)A,(__nv_bfloat16*)B,(__nv_bfloat16*)C,n,scalar);

    switch (op) {
        case ElemOp::Add:  has_b ? LAUNCH(ElemOp::Add,true) : LAUNCH(ElemOp::Add,false); break;
        case ElemOp::Relu: has_b ? LAUNCH(ElemOp::Relu,true) : LAUNCH(ElemOp::Relu,false); break;
        case ElemOp::Gelu: has_b ? LAUNCH(ElemOp::Gelu,true) : LAUNCH(ElemOp::Gelu,false); break;
        case ElemOp::Silu: has_b ? LAUNCH(ElemOp::Silu,true) : LAUNCH(ElemOp::Silu,false); break;
        case ElemOp::Sigmoid: LAUNCH(ElemOp::Sigmoid,false); break;
        case ElemOp::Tanh:    LAUNCH(ElemOp::Tanh,false); break;
        case ElemOp::Exp:     LAUNCH(ElemOp::Exp,false); break;
        case ElemOp::Log:     LAUNCH(ElemOp::Log,false); break;
        case ElemOp::Mish:    LAUNCH(ElemOp::Mish,false); break;
        case ElemOp::HardSwish: LAUNCH(ElemOp::HardSwish,false); break;
        default: break;
    }
    #undef LAUNCH
    CUDA_CHECK(cudaGetLastError());
}

void launch_rms_norm(
    void* x, void* weight, void* out,
    int64_t B, int64_t S, int64_t D, float eps, cudaStream_t stream)
{
    dim3 grid(B * S);
    dim3 block(std::min((int64_t)32, D));
    rms_norm_fp32<<<grid, block, 0, stream>>>(
        (float*)x, (float*)weight, (float*)out, B, S, D, eps);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace pygo::kernels

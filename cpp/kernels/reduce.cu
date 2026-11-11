// Parallel reduction CUDA kernels: sum, mean, max, softmax
#include <cuda.h>
#include <cuda_fp16.h>
#include "pygo/cuda.h"
namespace pygo::kernels {

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

template<typename T>
__device__ __forceinline__ T warp_reduce_max(T val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val = max(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

template<typename T>
__global__ void row_sum_kernel(const T* A, T* C, int64_t M, int64_t N) {
    __shared__ T smem[32];
    int64_t row = blockIdx.x;
    if (row >= M) return;
    T acc = T(0);
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) acc += A[row*N+j];
    acc = warp_reduce_sum(acc);
    if (threadIdx.x % 32 == 0) smem[threadIdx.x/32] = acc;
    __syncthreads();
    if (threadIdx.x == 0) {
        T total = T(0);
        for (int w = 0; w < (blockDim.x+31)/32; ++w) total += smem[w];
        C[row] = total;
    }
}

template<typename T>
__global__ void softmax_kernel(const T* X, T* Y, int64_t M, int64_t N) {
    __shared__ T smem[64];
    int64_t row = blockIdx.x;
    const T* x = X + row*N; T* y = Y + row*N;
    T mx = x[0];
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) mx = max(mx, x[j]);
    mx = warp_reduce_max(mx);
    if (threadIdx.x % 32 == 0) smem[threadIdx.x/32] = mx;
    __syncthreads();
    if (threadIdx.x == 0) { T g = smem[0]; for (int w=1;w<(blockDim.x+31)/32;++w) g=max(g,smem[w]); smem[0]=g; }
    __syncthreads();
    mx = smem[0];
    T sum = T(0);
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) { T e=exp(x[j]-mx); y[j]=e; sum+=e; }
    sum = warp_reduce_sum(sum);
    if (threadIdx.x % 32 == 0) smem[threadIdx.x/32] = sum;
    __syncthreads();
    if (threadIdx.x == 0) { T t=T(0); for (int w=0;w<(blockDim.x+31)/32;++w) t+=smem[w]; smem[0]=t; }
    __syncthreads();
    T inv = T(1)/smem[0];
    for (int64_t j = threadIdx.x; j < N; j += blockDim.x) y[j] *= inv;
}

void launch_row_sum(const void* A, void* C, int64_t M, int64_t N, DType dt, cudaStream_t s) {
    if (dt == DType::Float32)
        row_sum_kernel<float><<<M,256,0,s>>>((float*)A,(float*)C,M,N);
    CUDA_CHECK(cudaGetLastError());
}

void launch_softmax(const void* X, void* Y, int64_t M, int64_t N, DType dt, cudaStream_t s) {
    if (dt == DType::Float32)
        softmax_kernel<float><<<M,256,64*4,s>>>((float*)X,(float*)Y,M,N);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace pygo::kernels

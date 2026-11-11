#include "pygo/cuda.h"
#include <cuda_fp16.h>
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>
#include <float.h>

namespace pygo::cuda {

namespace {

enum class ReduceOp {
    Sum,
    Max,
    Min
};

template <ReduceOp Op>
__device__ inline float reduce_op(float a, float b) {
    if constexpr (Op == ReduceOp::Sum) return a + b;
    else if constexpr (Op == ReduceOp::Max) return fmaxf(a, b);
    else if constexpr (Op == ReduceOp::Min) return fminf(a, b);
}

template <ReduceOp Op>
__device__ inline float identity_val() {
    if constexpr (Op == ReduceOp::Sum) return 0.0f;
    else if constexpr (Op == ReduceOp::Max) return -FLT_MAX;
    else if constexpr (Op == ReduceOp::Min) return FLT_MAX;
}

template <ReduceOp Op>
__global__ void reduce_kernel(
    const __half* __restrict__ input,
    float* __restrict__ block_results,
    int64_t numel) {

    extern __shared__ float s_data[];

    int tid = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + threadIdx.x;

    float val = identity_val<Op>();
    if (idx < numel) {
        val = __half2float(input[idx]);
    }

    // Grid-stride loop for large inputs
    for (int i = idx + blockDim.x * gridDim.x; i < numel; i += blockDim.x * gridDim.x) {
        val = reduce_op<Op>(val, __half2float(input[i]));
    }

    s_data[tid] = val;
    __syncthreads();

    // Block reduction in shared memory
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] = reduce_op<Op>(s_data[tid], s_data[tid + s]);
        }
        __syncthreads();
    }

    if (tid == 0) {
        block_results[blockIdx.x] = s_data[0];
    }
}

template <ReduceOp Op>
Tensor reduce_dispatch(const Tensor& input, const std::vector<int64_t>& dims, bool keepdim) {
    if (!input.device().is_cuda() && !is_available())
        throw std::runtime_error("pygo::cuda::reduce: no CUDA device available");

    if (!dims.empty()) {
        throw std::runtime_error("pygo::cuda::reduce with specific dims is not yet implemented.");
    }

    int64_t numel = input.numel();
    if (numel == 0) {
        throw std::runtime_error("pygo::cuda::reduce on empty tensor");
    }

    Tensor input_c = input.contiguous();
    size_t bytes = numel * sizeof(__half);
    const __half* d_input = static_cast<const __half*>(to_device(input_c.data<void>(), bytes));

    int threads = 256;
    int blocks = (int)((numel + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;

    float* d_block_results = nullptr;
    try {
        PYGO_CUDA_CHECK(cudaMalloc(&d_block_results, blocks * sizeof(float)));

        size_t smem_size = threads * sizeof(float);
        reduce_kernel<Op><<<blocks, threads, smem_size>>>(
            d_input, d_block_results, numel
        );

        PYGO_CUDA_CHECK(cudaGetLastError());
        PYGO_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<float> h_block_results(blocks);
        PYGO_CUDA_CHECK(cudaMemcpy(h_block_results.data(), d_block_results, blocks * sizeof(float), cudaMemcpyDeviceToHost));

        float final_val = identity_val<Op>();
        for (int i = 0; i < blocks; ++i) {
            final_val = reduce_op<Op>(final_val, h_block_results[i]);
        }

        if constexpr (Op == ReduceOp::Sum) {
            // handled normally
        }

        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(input.ndim(), 1) : std::vector<int64_t>{1};
        Tensor out = empty(Shape(out_shape), DType::Float16, Device::CPU);
        *out.mutable_data<__half>() = __float2half(final_val);

        free_device(const_cast<__half*>(d_input));
        cudaFree(d_block_results);

        return out;
    } catch (...) {
        free_device(const_cast<__half*>(d_input));
        if (d_block_results) cudaFree(d_block_results);
        throw;
    }
}

} // anonymous namespace

Tensor reduce_sum_fp16(const Tensor& input, const std::vector<int64_t>& dims, bool keepdim) {
    return reduce_dispatch<ReduceOp::Sum>(input, dims, keepdim);
}

Tensor reduce_mean_fp16(const Tensor& input, const std::vector<int64_t>& dims, bool keepdim) {
    Tensor sum_tensor = reduce_sum_fp16(input, dims, keepdim);
    int64_t numel = input.numel();
    if (numel == 0) return sum_tensor;

    float val = __half2float(*sum_tensor.data<__half>());
    val /= static_cast<float>(numel);
    *sum_tensor.mutable_data<__half>() = __float2half(val);
    return sum_tensor;
}

Tensor reduce_max_fp16(const Tensor& input, const std::vector<int64_t>& dims, bool keepdim) {
    return reduce_dispatch<ReduceOp::Max>(input, dims, keepdim);
}

Tensor reduce_min_fp16(const Tensor& input, const std::vector<int64_t>& dims, bool keepdim) {
    return reduce_dispatch<ReduceOp::Min>(input, dims, keepdim);
}

} // namespace pygo::cuda
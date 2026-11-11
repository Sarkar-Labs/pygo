#include "pygo/matmul.h"
#include <stdexcept>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T>
void matmul_cpu_kernel(const T* a, const T* b, T* c, int64_t M, int64_t K, int64_t N) {
    #pragma omp parallel for collapse(2) schedule(static) if(M * N > 1024)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            T sum = static_cast<T>(0);
            #pragma omp simd reduction(+:sum)
            for (int64_t k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            c[i * N + j] = sum;
        }
    }
}

} // namespace detail

Tensor matmul(const Tensor& input1, const Tensor& input2) {
    if (!input1.device().is_cpu() || !input2.device().is_cpu()) {
        throw std::runtime_error("pygo::matmul is currently supported only on CPU.");
    }

    if (input1.dtype() != input2.dtype()) {
        throw std::invalid_argument("pygo::matmul expects operands to have the same dtype.");
    }

    if (input1.shape().ndim() != 2 || input2.shape().ndim() != 2) {
        throw std::invalid_argument("pygo::matmul requires 2D tensors.");
    }

    int64_t M = input1.shape()[0];
    int64_t K1 = input1.shape()[1];
    int64_t K2 = input2.shape()[0];
    int64_t N = input2.shape()[1];

    if (K1 != K2) {
        throw std::invalid_argument("pygo::matmul inner dimensions must match.");
    }

    auto output = zeros({M, N}, input1.dtype(), input1.device());

    if (input1.dtype() == DType::Float32) {
        detail::matmul_cpu_kernel<float>(
            input1.data<float>(), input2.data<float>(), output.data<float>(), M, K1, N
        );
    } else if (input1.dtype() == DType::Float64) {
        detail::matmul_cpu_kernel<double>(
            input1.data<double>(), input2.data<double>(), output.data<double>(), M, K1, N
        );
    } else if (input1.dtype() == DType::Int32) {
        detail::matmul_cpu_kernel<int32_t>(
            input1.data<int32_t>(), input2.data<int32_t>(), output.data<int32_t>(), M, K1, N
        );
    } else if (input1.dtype() == DType::Int64) {
        detail::matmul_cpu_kernel<int64_t>(
            input1.data<int64_t>(), input2.data<int64_t>(), output.data<int64_t>(), M, K1, N
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::matmul.");
    }

    return output;
}

} // namespace pygo
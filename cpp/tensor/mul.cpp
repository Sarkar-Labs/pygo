#include "pygo/mul.h"
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
void mul_cpu_kernel(const T* a, const T* b, T* dst, int64_t numel) {
    #pragma omp parallel for simd schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        dst[i] = a[i] * b[i];
    }
}

template <typename T>
void mul_inplace_cpu_kernel(T* a, const T* b, int64_t numel) {
    #pragma omp parallel for simd schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        a[i] *= b[i];
    }
}

} // namespace detail

Tensor mul(const Tensor& input1, const Tensor& input2) {
    if (!input1.device().is_cpu() || !input2.device().is_cpu()) {
        throw std::runtime_error("pygo::mul is currently supported only on CPU.");
    }

    if (input1.dtype() != input2.dtype()) {
        throw std::invalid_argument("pygo::mul expects operands to have the same dtype.");
    }

    if (input1.shape() != input2.shape()) {
        throw std::invalid_argument("pygo::mul expects operands to have the same shape.");
    }

    auto output = zeros(input1.shape(), input1.dtype(), input1.device());
    int64_t numel = input1.numel();

    if (input1.dtype() == DType::Float32) {
        detail::mul_cpu_kernel<float>(input1.data<float>(), input2.data<float>(), output.data<float>(), numel);
    } else if (input1.dtype() == DType::Float64) {
        detail::mul_cpu_kernel<double>(input1.data<double>(), input2.data<double>(), output.data<double>(), numel);
    } else if (input1.dtype() == DType::Int32) {
        detail::mul_cpu_kernel<int32_t>(input1.data<int32_t>(), input2.data<int32_t>(), output.data<int32_t>(), numel);
    } else if (input1.dtype() == DType::Int64) {
        detail::mul_cpu_kernel<int64_t>(input1.data<int64_t>(), input2.data<int64_t>(), output.data<int64_t>(), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::mul.");
    }

    return output;
}

void mul_(Tensor& input1, const Tensor& input2) {
    if (!input1.device().is_cpu() || !input2.device().is_cpu()) {
        throw std::runtime_error("pygo::mul_ is currently supported only on CPU.");
    }

    if (input1.dtype() != input2.dtype()) {
        throw std::invalid_argument("pygo::mul_ expects operands to have the same dtype.");
    }

    if (input1.shape() != input2.shape()) {
        throw std::invalid_argument("pygo::mul_ expects operands to have the same shape.");
    }

    int64_t numel = input1.numel();

    if (input1.dtype() == DType::Float32) {
        detail::mul_inplace_cpu_kernel<float>(input1.data<float>(), input2.data<float>(), numel);
    } else if (input1.dtype() == DType::Float64) {
        detail::mul_inplace_cpu_kernel<double>(input1.data<double>(), input2.data<double>(), numel);
    } else if (input1.dtype() == DType::Int32) {
        detail::mul_inplace_cpu_kernel<int32_t>(input1.data<int32_t>(), input2.data<int32_t>(), numel);
    } else if (input1.dtype() == DType::Int64) {
        detail::mul_inplace_cpu_kernel<int64_t>(input1.data<int64_t>(), input2.data<int64_t>(), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::mul_.");
    }
}

} // namespace pygo
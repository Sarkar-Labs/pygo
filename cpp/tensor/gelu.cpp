#include "pygo/gelu.h"
#include <cmath>
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
void gelu_cpu_kernel(const T* src, T* dst, int64_t numel) {
    const T inv_sqrt2 = static_cast<T>(0.70710678118654752440);
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1.0);

    #pragma omp parallel for simd schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        T val = src[i];
        dst[i] = half * val * (one + std::erf(val * inv_sqrt2));
    }
}

template <typename T>
void gelu_inplace_cpu_kernel(T* data, int64_t numel) {
    const T inv_sqrt2 = static_cast<T>(0.70710678118654752440);
    const T half = static_cast<T>(0.5);
    const T one = static_cast<T>(1.0);

    #pragma omp parallel for simd schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        T val = data[i];
        data[i] = half * val * (one + std::erf(val * inv_sqrt2));
    }
}

} // namespace detail

Tensor gelu(const Tensor& input) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::gelu is currently supported only on CPU.");
    }

    auto output = zeros(input.shape(), input.dtype(), input.device());
    int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        detail::gelu_cpu_kernel<float>(input.data<float>(), output.data<float>(), numel);
    } else if (input.dtype() == DType::Float64) {
        detail::gelu_cpu_kernel<double>(input.data<double>(), output.data<double>(), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::gelu. Expected floating-point type.");
    }

    return output;
}

void gelu_(Tensor& input) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::gelu_ is currently supported only on CPU.");
    }

    int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        detail::gelu_inplace_cpu_kernel<float>(input.data<float>(), numel);
    } else if (input.dtype() == DType::Float64) {
        detail::gelu_inplace_cpu_kernel<double>(input.data<double>(), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::gelu_. Expected floating-point type.");
    }
}

} // namespace pygo
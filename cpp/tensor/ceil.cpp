#include "pygo/ceil.h"
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
void ceil_cpu_kernel(const T* src, T* dst, int64_t numel) {
    #pragma omp parallel for schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        dst[i] = std::ceil(src[i]);
    }
}

template <typename T>
void ceil_inplace_cpu_kernel(T* data, int64_t numel) {
    #pragma omp parallel for schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        data[i] = std::ceil(data[i]);
    }
}

} // namespace detail

Tensor ceil(const Tensor& input) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::ceil is currently supported only on CPU.");
    }

    auto output = zeros(input.shape(), input.dtype(), input.device());
    int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        detail::ceil_cpu_kernel<float>(input.data<float>(), output.data<float>(), numel);
    } else if (input.dtype() == DType::Float64) {
        detail::ceil_cpu_kernel<double>(input.data<double>(), output.data<double>(), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::ceil. Expected floating-point type.");
    }

    return output;
}

void ceil_(Tensor& input) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::ceil_ is currently supported only on CPU.");
    }

    int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        detail::ceil_inplace_cpu_kernel<float>(input.data<float>(), numel);
    } else if (input.dtype() == DType::Float64) {
        detail::ceil_inplace_cpu_kernel<double>(input.data<double>(), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::ceil_. Expected floating-point type.");
    }
}

} // namespace pygo
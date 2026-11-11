#include "pygo/div.h"
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
void div_tensor_cpu_kernel(const T* a, const T* b, T* dst, int64_t numel) {
    #pragma omp parallel for simd schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        dst[i] = a[i] / b[i];
    }
}

template <typename T>
void div_scalar_cpu_kernel(const T* src, T scalar, T* dst, int64_t numel) {
    #pragma omp parallel for simd schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        dst[i] = src[i] / scalar;
    }
}

template <typename T>
void div_tensor_inplace_cpu_kernel(T* a, const T* b, int64_t numel) {
    #pragma omp parallel for simd schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        a[i] /= b[i];
    }
}

template <typename T>
void div_scalar_inplace_cpu_kernel(T* data, T scalar, int64_t numel) {
    #pragma omp parallel for simd schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        data[i] /= scalar;
    }
}

} // namespace detail

Tensor div(const Tensor& input, const Tensor& other) {
    if (!input.device().is_cpu() || !other.device().is_cpu()) {
        throw std::runtime_error("pygo::div is currently supported only on CPU.");
    }

    if (input.dtype() != other.dtype()) {
        throw std::invalid_argument("pygo::div expects inputs to have the same dtype.");
    }

    if (input.shape() != other.shape()) {
        throw std::invalid_argument("pygo::div expects inputs to have the same shape.");
    }

    auto output = zeros(input.shape(), input.dtype(), input.device());
    int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        detail::div_tensor_cpu_kernel<float>(input.data<float>(), other.data<float>(), output.data<float>(), numel);
    } else if (input.dtype() == DType::Float64) {
        detail::div_tensor_cpu_kernel<double>(input.data<double>(), other.data<double>(), output.data<double>(), numel);
    } else if (input.dtype() == DType::Int32) {
        detail::div_tensor_cpu_kernel<int32_t>(input.data<int32_t>(), other.data<int32_t>(), output.data<int32_t>(), numel);
    } else if (input.dtype() == DType::Int64) {
        detail::div_tensor_cpu_kernel<int64_t>(input.data<int64_t>(), other.data<int64_t>(), output.data<int64_t>(), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::div.");
    }

    return output;
}

Tensor div(const Tensor& input, double scalar) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::div is currently supported only on CPU.");
    }

    auto output = zeros(input.shape(), input.dtype(), input.device());
    int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        detail::div_scalar_cpu_kernel<float>(input.data<float>(), static_cast<float>(scalar), output.data<float>(), numel);
    } else if (input.dtype() == DType::Float64) {
        detail::div_scalar_cpu_kernel<double>(input.data<double>(), static_cast<double>(scalar), output.data<double>(), numel);
    } else if (input.dtype() == DType::Int32) {
        detail::div_scalar_cpu_kernel<int32_t>(input.data<int32_t>(), static_cast<int32_t>(scalar), output.data<int32_t>(), numel);
    } else if (input.dtype() == DType::Int64) {
        detail::div_scalar_cpu_kernel<int64_t>(input.data<int64_t>(), static_cast<int64_t>(scalar), output.data<int64_t>(), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::div.");
    }

    return output;
}

void div_(Tensor& input, const Tensor& other) {
    if (!input.device().is_cpu() || !other.device().is_cpu()) {
        throw std::runtime_error("pygo::div_ is currently supported only on CPU.");
    }

    if (input.dtype() != other.dtype()) {
        throw std::invalid_argument("pygo::div_ expects inputs to have the same dtype.");
    }

    if (input.shape() != other.shape()) {
        throw std::invalid_argument("pygo::div_ expects inputs to have the same shape.");
    }

    int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        detail::div_tensor_inplace_cpu_kernel<float>(input.data<float>(), other.data<float>(), numel);
    } else if (input.dtype() == DType::Float64) {
        detail::div_tensor_inplace_cpu_kernel<double>(input.data<double>(), other.data<double>(), numel);
    } else if (input.dtype() == DType::Int32) {
        detail::div_tensor_inplace_cpu_kernel<int32_t>(input.data<int32_t>(), other.data<int32_t>(), numel);
    } else if (input.dtype() == DType::Int64) {
        detail::div_tensor_inplace_cpu_kernel<int64_t>(input.data<int64_t>(), other.data<int64_t>(), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::div_.");
    }
}

void div_(Tensor& input, double scalar) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::div_ is currently supported only on CPU.");
    }

    int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        detail::div_scalar_inplace_cpu_kernel<float>(input.data<float>(), static_cast<float>(scalar), numel);
    } else if (input.dtype() == DType::Float64) {
        detail::div_scalar_inplace_cpu_kernel<double>(input.data<double>(), static_cast<double>(scalar), numel);
    } else if (input.dtype() == DType::Int32) {
        detail::div_scalar_inplace_cpu_kernel<int32_t>(input.data<int32_t>(), static_cast<int32_t>(scalar), numel);
    } else if (input.dtype() == DType::Int64) {
        detail::div_scalar_inplace_cpu_kernel<int64_t>(input.data<int64_t>(), static_cast<int64_t>(scalar), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::div_.");
    }
}

} // namespace pygo
#include "pygo/clip.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T>
void clip_cpu_kernel(const T* src, T* dst, int64_t numel, double min_val, double max_val) {
    T low = static_cast<T>(min_val);
    T high = static_cast<T>(max_val);

    #pragma omp parallel for schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        dst[i] = std::max(low, std::min(src[i], high));
    }
}

template <typename T>
void clip_inplace_cpu_kernel(T* data, int64_t numel, double min_val, double max_val) {
    T low = static_cast<T>(min_val);
    T high = static_cast<T>(max_val);

    #pragma omp parallel for schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        data[i] = std::max(low, std::min(data[i], high));
    }
}

} // namespace detail

Tensor clip(const Tensor& input, double min_val, double max_val) {
    if (min_val > max_val) {
        throw std::invalid_argument("pygo::clip min_val cannot be greater than max_val.");
    }

    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::clip is currently supported only on CPU.");
    }

    auto output = zeros(input.shape(), input.dtype(), input.device());
    int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        detail::clip_cpu_kernel<float>(input.data<float>(), output.data<float>(), numel, min_val, max_val);
    } else if (input.dtype() == DType::Float64) {
        detail::clip_cpu_kernel<double>(input.data<double>(), output.data<double>(), numel, min_val, max_val);
    } else if (input.dtype() == DType::Int32) {
        detail::clip_cpu_kernel<int32_t>(input.data<int32_t>(), output.data<int32_t>(), numel, min_val, max_val);
    } else if (input.dtype() == DType::Int64) {
        detail::clip_cpu_kernel<int64_t>(input.data<int64_t>(), output.data<int64_t>(), numel, min_val, max_val);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::clip.");
    }

    return output;
}

void clip_(Tensor& input, double min_val, double max_val) {
    if (min_val > max_val) {
        throw std::invalid_argument("pygo::clip_ min_val cannot be greater than max_val.");
    }

    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::clip_ is currently supported only on CPU.");
    }

    int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        detail::clip_inplace_cpu_kernel<float>(input.data<float>(), numel, min_val, max_val);
    } else if (input.dtype() == DType::Float64) {
        detail::clip_inplace_cpu_kernel<double>(input.data<double>(), numel, min_val, max_val);
    } else if (input.dtype() == DType::Int32) {
        detail::clip_inplace_cpu_kernel<int32_t>(input.data<int32_t>(), numel, min_val, max_val);
    } else if (input.dtype() == DType::Int64) {
        detail::clip_inplace_cpu_kernel<int64_t>(input.data<int64_t>(), numel, min_val, max_val);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::clip_.");
    }
}

} // namespace pygo
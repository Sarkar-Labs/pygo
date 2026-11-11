#include "pygo/outer.h"
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
void outer_cpu_kernel(const T* a, const T* b, T* dst, int64_t n, int64_t m) {
    #pragma omp parallel for collapse(2) schedule(static) if(n * m > 1024)
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            dst[i * m + j] = a[i] * b[j];
        }
    }
}

} // namespace detail

Tensor outer(const Tensor& vec1, const Tensor& vec2) {
    if (!vec1.device().is_cpu() || !vec2.device().is_cpu()) {
        throw std::runtime_error("pygo::outer is currently supported only on CPU.");
    }

    if (vec1.dtype() != vec2.dtype()) {
        throw std::invalid_argument("pygo::outer expects operands to have the same dtype.");
    }

    if (vec1.shape().ndim() != 1 || vec2.shape().ndim() != 1) {
        throw std::invalid_argument("pygo::outer requires 1D tensors.");
    }

    int64_t n = vec1.shape()[0];
    int64_t m = vec2.shape()[0];

    auto output = zeros({n, m}, vec1.dtype(), vec1.device());

    if (vec1.dtype() == DType::Float32) {
        detail::outer_cpu_kernel<float>(vec1.data<float>(), vec2.data<float>(), output.data<float>(), n, m);
    } else if (vec1.dtype() == DType::Float64) {
        detail::outer_cpu_kernel<double>(vec1.data<double>(), vec2.data<double>(), output.data<double>(), n, m);
    } else if (vec1.dtype() == DType::Int32) {
        detail::outer_cpu_kernel<int32_t>(vec1.data<int32_t>(), vec2.data<int32_t>(), output.data<int32_t>(), n, m);
    } else if (vec1.dtype() == DType::Int64) {
        detail::outer_cpu_kernel<int64_t>(vec1.data<int64_t>(), vec2.data<int64_t>(), output.data<int64_t>(), n, m);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::outer.");
    }

    return output;
}

} // namespace pygo
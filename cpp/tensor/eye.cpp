#include "pygo/eye.h"
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
void eye_cpu_kernel(T* ptr, int64_t n, int64_t m) {
    #pragma omp parallel for schedule(static) if(n > 64)
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < m; ++j) {
            ptr[i * m + j] = (i == j) ? static_cast<T>(1) : static_cast<T>(0);
        }
    }
}

} // namespace detail

Tensor eye(int64_t n, int64_t m, DType dtype, const Device& device) {
    if (!device.is_cpu()) {
        throw std::runtime_error("pygo::eye is currently supported only on CPU.");
    }

    if (n < 0) {
        throw std::invalid_argument("pygo::eye number of rows must be non-negative.");
    }

    if (m == -1) {
        m = n;
    }

    if (m < 0) {
        throw std::invalid_argument("pygo::eye number of columns must be non-negative.");
    }

    auto output = zeros({n, m}, dtype, device);

    if (dtype == DType::Float32) {
        detail::eye_cpu_kernel<float>(output.data<float>(), n, m);
    } else if (dtype == DType::Float64) {
        detail::eye_cpu_kernel<double>(output.data<double>(), n, m);
    } else if (dtype == DType::Int32) {
        detail::eye_cpu_kernel<int32_t>(output.data<int32_t>(), n, m);
    } else if (dtype == DType::Int64) {
        detail::eye_cpu_kernel<int64_t>(output.data<int64_t>(), n, m);
    } else if (dtype == DType::Bool) {
        detail::eye_cpu_kernel<bool>(output.data<bool>(), n, m);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::eye.");
    }

    return output;
}

} // namespace pygo
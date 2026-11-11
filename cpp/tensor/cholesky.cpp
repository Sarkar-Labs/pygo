#include "pygo/cholesky.h"
#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace pygo {

namespace detail {

template <typename T>
void cholesky_cpu_kernel(const T* src, T* dst, int64_t n, bool upper) {
    for (int64_t i = 0; i < n * n; ++i) {
        dst[i] = static_cast<T>(0);
    }

    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j <= i; ++j) {
            T sum = static_cast<T>(0);
            for (int64_t k = 0; k < j; ++k) {
                sum += dst[i * n + k] * dst[j * n + k];
            }

            if (i == j) {
                T val = src[i * n + i] - sum;
                if (val <= static_cast<T>(0)) {
                    throw std::runtime_error("pygo::cholesky input matrix is not positive-definite.");
                }
                dst[i * n + j] = std::sqrt(val);
            } else {
                dst[i * n + j] = (src[i * n + j] - sum) / dst[j * n + j];
            }
        }
    }

    if (upper) {
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = i + 1; j < n; ++j) {
                dst[i * n + j] = dst[j * n + i];
                dst[j * n + i] = static_cast<T>(0);
            }
        }
    }
}

template <typename T>
void batch_cholesky_cpu(const T* src, T* dst, int64_t batch_size, int64_t n, bool upper) {
    int64_t stride = n * n;

    #pragma omp parallel for schedule(static) if(batch_size > 1)
    for (int64_t b = 0; b < batch_size; ++b) {
        cholesky_cpu_kernel<T>(src + b * stride, dst + b * stride, n, upper);
    }
}

} // namespace detail

Tensor cholesky(const Tensor& input, bool upper) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::cholesky is currently supported only on CPU.");
    }

    const auto& shape = input.shape();
    size_t ndim = shape.size();

    if (ndim < 2) {
        throw std::invalid_argument("pygo::cholesky requires tensor with at least 2 dimensions.");
    }

    int64_t n = shape[ndim - 1];
    int64_t m = shape[ndim - 2];

    if (n != m) {
        throw std::invalid_argument("pygo::cholesky matrix must be square.");
    }

    int64_t batch_size = 1;
    for (size_t i = 0; i < ndim - 2; ++i) {
        batch_size *= shape[i];
    }

    auto output = zeros(Shape(shape), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        detail::batch_cholesky_cpu<float>(input.data<float>(), output.data<float>(), batch_size, n, upper);
    } else if (input.dtype() == DType::Float64) {
        detail::batch_cholesky_cpu<double>(input.data<double>(), output.data<double>(), batch_size, n, upper);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::cholesky. Expected Float32 or Float64.");
    }

    return output;
}

} // namespace pygo
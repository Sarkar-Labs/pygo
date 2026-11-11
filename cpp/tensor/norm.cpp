#include "pygo/norm.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace pygo {

namespace detail {

template <typename T>
T norm_cpu_kernel(const T* src, int64_t numel, double p) {
    T sum = static_cast<T>(0);
    if (p == 2.0) {
        #pragma omp parallel for simd reduction(+:sum) if(numel > 65536)
        for (int64_t i = 0; i < numel; ++i) {
            T val = src[i];
            sum += val * val;
        }
        return std::sqrt(sum);
    } else if (p == 1.0) {
        #pragma omp parallel for simd reduction(+:sum) if(numel > 65536)
        for (int64_t i = 0; i < numel; ++i) {
            sum += std::abs(src[i]);
        }
        return sum;
    } else {
        #pragma omp parallel for simd reduction(+:sum) if(numel > 65536)
        for (int64_t i = 0; i < numel; ++i) {
            sum += std::pow(std::abs(src[i]), static_cast<T>(p));
        }
        return std::pow(sum, static_cast<T>(1.0 / p));
    }
}

} // namespace detail

Tensor norm(const Tensor& input, double p) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::norm is currently supported only on CPU.");
    }

    int64_t numel = input.numel();
    auto output = zeros({}, input.dtype(), input.device());

    if (numel == 0) {
        return output;
    }

    if (input.dtype() == DType::Float32) {
        float res = detail::norm_cpu_kernel<float>(input.data<float>(), numel, p);
        *output.data<float>() = res;
    } else if (input.dtype() == DType::Float64) {
        double res = detail::norm_cpu_kernel<double>(input.data<double>(), numel, p);
        *output.data<double>() = res;
    } else {
        throw std::runtime_error("Unsupported DType for pygo::norm. Expected floating-point type.");
    }

    return output;
}

} // namespace pygo
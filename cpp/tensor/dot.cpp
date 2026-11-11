#include "pygo/dot.h"
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T>
T dot_cpu_kernel(const T* a, const T* b, int64_t numel) {
    T sum = static_cast<T>(0);

    #pragma omp parallel for reduction(+:sum) schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

} // namespace detail

Tensor dot(const Tensor& input, const Tensor& other) {
    if (!input.device().is_cpu() || !other.device().is_cpu()) {
        throw std::runtime_error("pygo::dot is currently supported only on CPU.");
    }

    if (input.dtype() != other.dtype()) {
        throw std::invalid_argument("pygo::dot expects inputs to have the same dtype.");
    }

    if (input.numel() != other.numel()) {
        throw std::invalid_argument("pygo::dot expects inputs to have the same number of elements.");
    }

    const auto& in_shape = input.shape();
    const auto& oth_shape = other.shape();

    if (in_shape.size() != 1 || oth_shape.size() != 1) {
        throw std::invalid_argument("pygo::dot expects 1D tensors (vectors).");
    }

    if (in_shape[0] != oth_shape[0]) {
        throw std::invalid_argument("pygo::dot expects vectors of matching lengths.");
    }

    int64_t numel = input.numel();
    auto output = zeros({}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        float val = detail::dot_cpu_kernel<float>(input.data<float>(), other.data<float>(), numel);
        *output.data<float>() = val;
    } else if (input.dtype() == DType::Float64) {
        double val = detail::dot_cpu_kernel<double>(input.data<double>(), other.data<double>(), numel);
        *output.data<double>() = val;
    } else if (input.dtype() == DType::Int32) {
        int32_t val = detail::dot_cpu_kernel<int32_t>(input.data<int32_t>(), other.data<int32_t>(), numel);
        *output.data<int32_t>() = val;
    } else if (input.dtype() == DType::Int64) {
        int64_t val = detail::dot_cpu_kernel<int64_t>(input.data<int64_t>(), other.data<int64_t>(), numel);
        *output.data<int64_t>() = val;
    } else {
        throw std::runtime_error("Unsupported DType for pygo::dot.");
    }

    return output;
}

} // namespace pygo
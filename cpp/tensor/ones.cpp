#include "pygo/ones.h"
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T>
void ones_cpu_kernel(T* data, int64_t numel) {
    #pragma omp parallel for simd schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        data[i] = static_cast<T>(1);
    }
}

} // namespace detail

Tensor ones(const std::vector<int64_t>& shape, DType dtype, const Device& device) {
    if (!device.is_cpu()) {
        throw std::runtime_error("pygo::ones is currently supported only on CPU.");
    }

    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("pygo::ones tensor shape dimensions must be non-negative.");
        }
    }

    auto output = zeros(Shape(shape), dtype, device);
    int64_t numel = output.numel();

    if (numel == 0) {
        return output;
    }

    if (dtype == DType::Float32) {
        detail::ones_cpu_kernel<float>(output.data<float>(), numel);
    } else if (dtype == DType::Float64) {
        detail::ones_cpu_kernel<double>(output.data<double>(), numel);
    } else if (dtype == DType::Int32) {
        detail::ones_cpu_kernel<int32_t>(output.data<int32_t>(), numel);
    } else if (dtype == DType::Int64) {
        detail::ones_cpu_kernel<int64_t>(output.data<int64_t>(), numel);
    } else if (dtype == DType::Bool) {
        detail::ones_cpu_kernel<bool>(output.data<bool>(), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::ones.");
    }

    return output;
}

} // namespace pygo
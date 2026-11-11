#include "pygo/linspace.h"
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T>
void linspace_cpu_kernel(T* data, double start, double step_size, int64_t steps) {
    #pragma omp parallel for simd schedule(static) if(steps > 65536)
    for (int64_t i = 0; i < steps; ++i) {
        data[i] = static_cast<T>(start + static_cast<double>(i) * step_size);
    }
}

} // namespace detail

Tensor linspace(double start, double stop, int64_t steps, DType dtype, const Device& device) {
    if (!device.is_cpu()) {
        throw std::runtime_error("pygo::linspace is currently supported only on CPU.");
    }

    if (steps < 0) {
        throw std::invalid_argument("pygo::linspace number of steps must be non-negative.");
    }

    auto output = zeros({steps}, dtype, device);

    if (steps == 0) {
        return output;
    }

    double step_size = (steps == 1) ? 0.0 : (stop - start) / static_cast<double>(steps - 1);

    if (dtype == DType::Float32) {
        detail::linspace_cpu_kernel<float>(output.data<float>(), start, step_size, steps);
    } else if (dtype == DType::Float64) {
        detail::linspace_cpu_kernel<double>(output.data<double>(), start, step_size, steps);
    } else if (dtype == DType::Int32) {
        detail::linspace_cpu_kernel<int32_t>(output.data<int32_t>(), start, step_size, steps);
    } else if (dtype == DType::Int64) {
        detail::linspace_cpu_kernel<int64_t>(output.data<int64_t>(), start, step_size, steps);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::linspace.");
    }

    return output;
}

} // namespace pygo
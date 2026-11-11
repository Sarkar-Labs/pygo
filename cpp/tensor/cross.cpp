#include "pygo/cross.h"
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T>
void cross_cpu_kernel(const T* a, const T* b, T* out, 
                      int64_t outer_size, int64_t inner_size) {
    int64_t stride = inner_size;

    #pragma omp parallel for collapse(2) schedule(static) if(outer_size * inner_size > 1024)
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t i = 0; i < inner_size; ++i) {
            int64_t base_idx = o * 3 * inner_size + i;
            
            int64_t idx0 = base_idx;
            int64_t idx1 = base_idx + stride;
            int64_t idx2 = base_idx + 2 * stride;

            T a0 = a[idx0];
            T a1 = a[idx1];
            T a2 = a[idx2];

            T b0 = b[idx0];
            T b1 = b[idx1];
            T b2 = b[idx2];

            out[idx0] = a1 * b2 - a2 * b1;
            out[idx1] = a2 * b0 - a0 * b2;
            out[idx2] = a0 * b1 - a1 * b0;
        }
    }
}

} // namespace detail

Tensor cross(const Tensor& input, const Tensor& other, int64_t dim) {
    if (!input.device().is_cpu() || !other.device().is_cpu()) {
        throw std::runtime_error("pygo::cross is currently supported only on CPU.");
    }

    if (input.dtype() != other.dtype()) {
        throw std::invalid_argument("pygo::cross expects inputs to have the same dtype.");
    }

    const auto& shape = input.shape();
    const auto& other_shape = other.shape();

    if (shape != other_shape) {
        throw std::invalid_argument("pygo::cross expects inputs to have the same shape.");
    }

    int64_t ndim = static_cast<int64_t>(shape.size());
    if (ndim == 0) {
        throw std::invalid_argument("pygo::cross cannot be performed on 0-dim tensors.");
    }

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("Dimension out of range in pygo::cross.");
    }

    if (shape[dim] != 3) {
        throw std::invalid_argument("pygo::cross dimension size must be 3.");
    }

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= shape[i];
    }

    auto output = zeros(Shape(shape), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        detail::cross_cpu_kernel<float>(
            input.data<float>(), other.data<float>(), output.data<float>(), 
            outer_size, inner_size
        );
    } else if (input.dtype() == DType::Float64) {
        detail::cross_cpu_kernel<double>(
            input.data<double>(), other.data<double>(), output.data<double>(), 
            outer_size, inner_size
        );
    } else if (input.dtype() == DType::Int32) {
        detail::cross_cpu_kernel<int32_t>(
            input.data<int32_t>(), other.data<int32_t>(), output.data<int32_t>(), 
            outer_size, inner_size
        );
    } else if (input.dtype() == DType::Int64) {
        detail::cross_cpu_kernel<int64_t>(
            input.data<int64_t>(), other.data<int64_t>(), output.data<int64_t>(), 
            outer_size, inner_size
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::cross.");
    }

    return output;
}

} // namespace pygo
#include "pygo/cumsum.h"
#include <stdexcept>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T>
void cumsum_cpu_kernel(const T* src, T* dst, int64_t outer_size, int64_t dim_size, int64_t inner_size) {
    if (inner_size == 1) {
        #pragma omp parallel for schedule(static) if(outer_size > 1024)
        for (int64_t o = 0; o < outer_size; ++o) {
            int64_t offset = o * dim_size;
            T sum = static_cast<T>(0);
            for (int64_t d = 0; d < dim_size; ++d) {
                sum += src[offset + d];
                dst[offset + d] = sum;
            }
        }
    } else {
        #pragma omp parallel for schedule(static) if(outer_size * inner_size > 4096)
        for (int64_t o = 0; o < outer_size; ++o) {
            int64_t base_offset = o * dim_size * inner_size;
            
            #pragma omp simd
            for (int64_t i = 0; i < inner_size; ++i) {
                dst[base_offset + i] = src[base_offset + i];
            }

            for (int64_t d = 1; d < dim_size; ++d) {
                int64_t prev_offset = base_offset + (d - 1) * inner_size;
                int64_t curr_offset = base_offset + d * inner_size;
                
                #pragma omp simd
                for (int64_t i = 0; i < inner_size; ++i) {
                    dst[curr_offset + i] = dst[prev_offset + i] + src[curr_offset + i];
                }
            }
        }
    }
}

} // namespace detail

Tensor cumsum(const Tensor& input, int64_t dim) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::cumsum is currently supported only on CPU.");
    }

    int64_t ndim = static_cast<int64_t>(input.shape().ndim());
    if (ndim == 0) {
        throw std::invalid_argument("pygo::cumsum cannot be performed on 0-dim tensor.");
    }

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("Dimension out of range in pygo::cumsum.");
    }

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.shape()[i];
    }

    int64_t dim_size = input.shape()[dim];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input.shape()[i];
    }

    auto output = zeros(input.shape(), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        detail::cumsum_cpu_kernel<float>(
            input.data<float>(), output.data<float>(),
            outer_size, dim_size, inner_size
        );
    } else if (input.dtype() == DType::Float64) {
        detail::cumsum_cpu_kernel<double>(
            input.data<double>(), output.data<double>(),
            outer_size, dim_size, inner_size
        );
    } else if (input.dtype() == DType::Int32) {
        detail::cumsum_cpu_kernel<int32_t>(
            input.data<int32_t>(), output.data<int32_t>(),
            outer_size, dim_size, inner_size
        );
    } else if (input.dtype() == DType::Int64) {
        detail::cumsum_cpu_kernel<int64_t>(
            input.data<int64_t>(), output.data<int64_t>(),
            outer_size, dim_size, inner_size
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::cumsum.");
    }

    return output;
}

} // namespace pygo
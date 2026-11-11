#include "pygo/diff.h"
#include <stdexcept>
#include <vector>
#include <cstring>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T>
void diff_cpu_kernel_n1(const T* src, T* dst, int64_t outer_size, int64_t dim_size, int64_t inner_size) {
    int64_t out_dim_size = dim_size - 1;

    if (inner_size == 1) {
        #pragma omp parallel for schedule(static) if(outer_size * out_dim_size > 4096)
        for (int64_t o = 0; o < outer_size; ++o) {
            int64_t src_base = o * dim_size;
            int64_t dst_base = o * out_dim_size;
            
            #pragma omp simd
            for (int64_t d = 0; d < out_dim_size; ++d) {
                dst[dst_base + d] = src[src_base + d + 1] - src[src_base + d];
            }
        }
    } else {
        #pragma omp parallel for collapse(2) schedule(static) if(outer_size * out_dim_size * inner_size > 4096)
        for (int64_t o = 0; o < outer_size; ++o) {
            for (int64_t d = 0; d < out_dim_size; ++d) {
                int64_t src_offset = o * dim_size * inner_size + d * inner_size;
                int64_t dst_offset = o * out_dim_size * inner_size + d * inner_size;
                
                #pragma omp simd
                for (int64_t i = 0; i < inner_size; ++i) {
                    dst[dst_offset + i] = src[src_offset + inner_size + i] - src[src_offset + i];
                }
            }
        }
    }
}

template <typename T>
void copy_tensor_data(const T* src, T* dst, int64_t numel) {
    std::memcpy(dst, src, numel * sizeof(T));
}

} // namespace detail

Tensor diff(const Tensor& input, int64_t n, int64_t dim) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::diff is currently supported only on CPU.");
    }

    if (n < 0) {
        throw std::invalid_argument("pygo::diff requires n >= 0.");
    }

    int64_t ndim = static_cast<int64_t>(input.shape().ndim());
    if (ndim == 0) {
        throw std::invalid_argument("pygo::diff cannot be performed on a 0-dimensional tensor.");
    }

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("Dimension out of range in pygo::diff.");
    }

    if (n == 0 || input.shape()[dim] == 0) {
        auto output = zeros(input.shape(), input.dtype(), input.device());
        int64_t numel = input.numel();
        if (numel > 0) {
            if (input.dtype() == DType::Float32) {
                detail::copy_tensor_data(input.data<float>(), output.data<float>(), numel);
            } else if (input.dtype() == DType::Float64) {
                detail::copy_tensor_data(input.data<double>(), output.data<double>(), numel);
            } else if (input.dtype() == DType::Int32) {
                detail::copy_tensor_data(input.data<int32_t>(), output.data<int32_t>(), numel);
            } else if (input.dtype() == DType::Int64) {
                detail::copy_tensor_data(input.data<int64_t>(), output.data<int64_t>(), numel);
            } else {
                throw std::runtime_error("Unsupported DType for pygo::diff.");
            }
        }
        return output;
    }

    if (input.shape()[dim] <= n) {
        std::vector<int64_t> out_shape = input.shape().dims;
        out_shape[dim] = 0;
        return zeros(Shape(out_shape), input.dtype(), input.device());
    }

    if (n > 1) {
        Tensor current = diff(input, 1, dim);
        for (int64_t i = 1; i < n; ++i) {
            current = diff(current, 1, dim);
        }
        return current;
    }

    std::vector<int64_t> out_shape = input.shape().dims;
    out_shape[dim] -= 1;

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.shape()[i];
    }

    int64_t dim_size = input.shape()[dim];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input.shape()[i];
    }

    auto output = zeros(Shape(out_shape), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        detail::diff_cpu_kernel_n1<float>(
            input.data<float>(), output.data<float>(),
            outer_size, dim_size, inner_size
        );
    } else if (input.dtype() == DType::Float64) {
        detail::diff_cpu_kernel_n1<double>(
            input.data<double>(), output.data<double>(),
            outer_size, dim_size, inner_size
        );
    } else if (input.dtype() == DType::Int32) {
        detail::diff_cpu_kernel_n1<int32_t>(
            input.data<int32_t>(), output.data<int32_t>(),
            outer_size, dim_size, inner_size
        );
    } else if (input.dtype() == DType::Int64) {
        detail::diff_cpu_kernel_n1<int64_t>(
            input.data<int64_t>(), output.data<int64_t>(),
            outer_size, dim_size, inner_size
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::diff.");
    }

    return output;
}

} // namespace pygo
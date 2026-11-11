#include "pygo/argmax.h"
#include <stdexcept>
#include <limits>
#include <algorithm>
#include <cmath>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T>
int64_t global_argmax_cpu(const T* data, int64_t numel) {
    if (numel <= 0) return 0;

    int64_t best_idx = 0;
    T max_val = data[0];

    #pragma omp parallel if(numel > 65536)
    {
        int64_t local_best_idx = 0;
        T local_max_val = -std::numeric_limits<T>::infinity();

        #pragma omp for schedule(static)
        for (int64_t i = 0; i < numel; ++i) {
            if (data[i] > local_max_val) {
                local_max_val = data[i];
                local_best_idx = i;
            }
        }

        #pragma omp critical
        {
            if (local_max_val > max_val || (local_max_val == max_val && local_best_idx < best_idx)) {
                max_val = local_max_val;
                best_idx = local_best_idx;
            }
        }
    }

    return best_idx;
}

template <typename T>
void dim_argmax_cpu(const T* input_data, int64_t* output_data,
                    const std::vector<int64_t>& shape, int64_t dim) {
    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= shape[i];
    }

    int64_t dim_size = shape[dim];

    int64_t inner_size = 1;
    for (size_t i = dim + 1; i < shape.size(); ++i) {
        inner_size *= shape[i];
    }

    int64_t slice_stride = dim_size * inner_size;

    #pragma omp parallel for collapse(2) schedule(static) if(outer_size * inner_size > 4096)
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t i = 0; i < inner_size; ++i) {
            const T* base_ptr = input_data + o * slice_stride + i;
            int64_t best_idx = 0;
            T max_val = base_ptr[0];

            for (int64_t d = 1; d < dim_size; ++d) {
                T val = base_ptr[d * inner_size];
                if (val > max_val) {
                    max_val = val;
                    best_idx = d;
                }
            }

            output_data[o * inner_size + i] = best_idx;
        }
    }
}

} // namespace detail

Tensor argmax(const Tensor& input, int64_t dim, bool keepdim) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::argmax is currently supported only on CPU.");
    }

    int64_t ndim = static_cast<int64_t>(input.shape().ndim());
    if (ndim == 0) {
        throw std::invalid_argument("pygo::argmax cannot be performed on 0-dim tensor.");
    }

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("Dimension out of range in pygo::argmax.");
    }

    std::vector<int64_t> out_shape;
    for (int64_t i = 0; i < ndim; ++i) {
        if (i == dim) {
            if (keepdim) out_shape.push_back(1);
        } else {
            out_shape.push_back(input.shape()[i]);
        }
    }

    auto output = zeros(Shape(out_shape), DType::Int64, input.device());

    if (input.dtype() == DType::Float32) {
        detail::dim_argmax_cpu<float>(input.data<float>(), output.data<int64_t>(), input.shape().dims, dim);
    } else if (input.dtype() == DType::Float64) {
        detail::dim_argmax_cpu<double>(input.data<double>(), output.data<int64_t>(), input.shape().dims, dim);
    } else if (input.dtype() == DType::Int32) {
        detail::dim_argmax_cpu<int32_t>(input.data<int32_t>(), output.data<int64_t>(), input.shape().dims, dim);
    } else if (input.dtype() == DType::Int64) {
        detail::dim_argmax_cpu<int64_t>(input.data<int64_t>(), output.data<int64_t>(), input.shape().dims, dim);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::argmax.");
    }

    return output;
}

Tensor argmax(const Tensor& input) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::argmax is currently supported only on CPU.");
    }

    int64_t numel = input.numel();
    if (numel == 0) {
        throw std::invalid_argument("pygo::argmax cannot be performed on empty tensor.");
    }

    auto output = zeros({1}, DType::Int64, input.device());
    int64_t idx = 0;

    if (input.dtype() == DType::Float32) {
        idx = detail::global_argmax_cpu<float>(input.data<float>(), numel);
    } else if (input.dtype() == DType::Float64) {
        idx = detail::global_argmax_cpu<double>(input.data<double>(), numel);
    } else if (input.dtype() == DType::Int32) {
        idx = detail::global_argmax_cpu<int32_t>(input.data<int32_t>(), numel);
    } else if (input.dtype() == DType::Int64) {
        idx = detail::global_argmax_cpu<int64_t>(input.data<int64_t>(), numel);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::argmax.");
    }

    output.data<int64_t>()[0] = idx;
    return output;
}

} // namespace pygo
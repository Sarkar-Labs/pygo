#include "pygo/index.h"
#include <stdexcept>
#include <vector>

namespace pygo {

namespace detail {

template <typename T, typename IndexT>
void index_select_cpu_kernel(const T* src, const IndexT* indices, T* dst,
                             int64_t outer_size, int64_t dim_size, int64_t inner_size,
                             int64_t index_size) {
    #pragma omp parallel for collapse(2) schedule(static) if(outer_size * index_size > 1024)
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t idx = 0; idx < index_size; ++idx) {
            IndexT i_val = indices[idx];
            if (i_val < 0) {
                i_val += dim_size;
            }
            if (i_val < 0 || i_val >= dim_size) {
                throw std::out_of_range("Index out of range in pygo::index_select.");
            }

            for (int64_t i = 0; i < inner_size; ++i) {
                int64_t src_idx = o * dim_size * inner_size + static_cast<int64_t>(i_val) * inner_size + i;
                int64_t dst_idx = o * index_size * inner_size + idx * inner_size + i;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

} // namespace detail

Tensor index_select(const Tensor& input, int64_t dim, const Tensor& index) {
    if (!input.device().is_cpu() || !index.device().is_cpu()) {
        throw std::runtime_error("pygo::index_select is currently supported only on CPU.");
    }

    int64_t ndim = static_cast<int64_t>(input.shape().ndim());
    if (ndim == 0) {
        throw std::invalid_argument("pygo::index_select cannot be performed on a 0-dimensional tensor.");
    }

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("Dimension out of range in pygo::index_select.");
    }

    if (index.shape().ndim() != 1) {
        throw std::invalid_argument("pygo::index_select requires a 1D index tensor.");
    }

    const auto& input_shape = input.shape();
    std::vector<int64_t> output_shape = input_shape.dims;
    output_shape[dim] = index.shape()[0];

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input_shape[i];
    }

    int64_t dim_size = input_shape[dim];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input_shape[i];
    }

    int64_t index_size = index.shape()[0];
    auto output = zeros(Shape(output_shape), input.dtype(), input.device());

    auto dispatch_index_select = [&](auto* dummy_in, auto* dummy_idx) {
        using T = std::remove_pointer_t<decltype(dummy_in)>;
        using IndexT = std::remove_pointer_t<decltype(dummy_idx)>;

        detail::index_select_cpu_kernel<T, IndexT>(
            input.data<T>(),
            index.data<IndexT>(),
            output.data<T>(),
            outer_size, dim_size, inner_size, index_size
        );
    };

    if (input.dtype() == DType::Float32) {
        if (index.dtype() == DType::Int32) {
            dispatch_index_select(static_cast<float*>(nullptr), static_cast<int32_t*>(nullptr));
        } else if (index.dtype() == DType::Int64) {
            dispatch_index_select(static_cast<float*>(nullptr), static_cast<int64_t*>(nullptr));
        } else {
            throw std::runtime_error("Unsupported index DType for pygo::index_select.");
        }
    } else if (input.dtype() == DType::Float64) {
        if (index.dtype() == DType::Int32) {
            dispatch_index_select(static_cast<double*>(nullptr), static_cast<int32_t*>(nullptr));
        } else if (index.dtype() == DType::Int64) {
            dispatch_index_select(static_cast<double*>(nullptr), static_cast<int64_t*>(nullptr));
        } else {
            throw std::runtime_error("Unsupported index DType for pygo::index_select.");
        }
    } else if (input.dtype() == DType::Int32) {
        if (index.dtype() == DType::Int32) {
            dispatch_index_select(static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr));
        } else if (index.dtype() == DType::Int64) {
            dispatch_index_select(static_cast<int32_t*>(nullptr), static_cast<int64_t*>(nullptr));
        } else {
            throw std::runtime_error("Unsupported index DType for pygo::index_select.");
        }
    } else if (input.dtype() == DType::Int64) {
        if (index.dtype() == DType::Int32) {
            dispatch_index_select(static_cast<int64_t*>(nullptr), static_cast<int32_t*>(nullptr));
        } else if (index.dtype() == DType::Int64) {
            dispatch_index_select(static_cast<int64_t*>(nullptr), static_cast<int64_t*>(nullptr));
        } else {
            throw std::runtime_error("Unsupported index DType for pygo::index_select.");
        }
    } else {
        throw std::runtime_error("Unsupported input DType for pygo::index_select.");
    }

    return output;
}

} // namespace pygo
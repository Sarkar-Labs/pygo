#include "pygo/gather.h"
#include <stdexcept>
#include <vector>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T, typename IndexT>
void gather_cpu_kernel(const T* src, const IndexT* index_data, T* dst,
                       const std::vector<int64_t>& input_shape,
                       const std::vector<int64_t>& index_shape,
                       int64_t dim, int64_t numel) {
    int64_t ndim = static_cast<int64_t>(input_shape.size());

    std::vector<int64_t> input_strides(ndim);
    int64_t stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        input_strides[i] = stride;
        stride *= input_shape[i];
    }

    std::vector<int64_t> index_strides(ndim);
    stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        index_strides[i] = stride;
        stride *= index_shape[i];
    }

    int64_t dim_size = input_shape[dim];

    #pragma omp parallel for schedule(static) if(numel > 4096)
    for (int64_t idx = 0; idx < numel; ++idx) {
        int64_t temp = idx;
        int64_t input_offset = 0;
        bool out_of_bounds = false;

        for (int64_t i = 0; i < ndim; ++i) {
            int64_t coord = temp / index_strides[i];
            temp %= index_strides[i];

            if (i == dim) {
                IndexT gathered_idx = index_data[idx];
                if (gathered_idx < 0) {
                    gathered_idx += dim_size;
                }
                if (gathered_idx < 0 || gathered_idx >= dim_size) {
                    out_of_bounds = true;
                    break;
                }
                input_offset += static_cast<int64_t>(gathered_idx) * input_strides[i];
            } else {
                input_offset += coord * input_strides[i];
            }
        }

        if (out_of_bounds) {
            dst[idx] = static_cast<T>(0);
        } else {
            dst[idx] = src[input_offset];
        }
    }
}

} // namespace detail

Tensor gather(const Tensor& input, int64_t dim, const Tensor& index) {
    if (!input.device().is_cpu() || !index.device().is_cpu()) {
        throw std::runtime_error("pygo::gather is currently supported only on CPU.");
    }

    int64_t ndim = static_cast<int64_t>(input.shape().ndim());
    int64_t index_ndim = static_cast<int64_t>(index.shape().ndim());

    if (ndim == 0) {
        throw std::invalid_argument("pygo::gather cannot be performed on a 0-dimensional tensor.");
    }

    if (ndim != index_ndim) {
        throw std::invalid_argument("pygo::gather requires index tensor to have the same number of dimensions as input.");
    }

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("Dimension out of range in pygo::gather.");
    }

    const auto& input_shape = input.shape().dims;
    const auto& index_shape = index.shape().dims;

    for (int64_t i = 0; i < ndim; ++i) {
        if (i != dim && index_shape[i] > input_shape[i]) {
            throw std::invalid_argument("pygo::gather index dimension exceeds input dimension for non-gather axes.");
        }
    }

    auto output = zeros(Shape(index_shape), input.dtype(), input.device());
    int64_t numel = index.numel();

    if (numel == 0) {
        return output;
    }

    auto dispatch_gather = [&](auto* dummy_in, auto* dummy_idx) {
        using T = std::remove_pointer_t<decltype(dummy_in)>;
        using IndexT = std::remove_pointer_t<decltype(dummy_idx)>;

        detail::gather_cpu_kernel<T, IndexT>(
            input.data<T>(),
            index.data<IndexT>(),
            output.data<T>(),
            input_shape,
            index_shape,
            dim,
            numel
        );
    };

    if (input.dtype() == DType::Float32) {
        if (index.dtype() == DType::Int32) {
            dispatch_gather(static_cast<float*>(nullptr), static_cast<int32_t*>(nullptr));
        } else if (index.dtype() == DType::Int64) {
            dispatch_gather(static_cast<float*>(nullptr), static_cast<int64_t*>(nullptr));
        } else {
            throw std::runtime_error("Unsupported index DType for pygo::gather. Expected Int32 or Int64.");
        }
    } else if (input.dtype() == DType::Float64) {
        if (index.dtype() == DType::Int32) {
            dispatch_gather(static_cast<double*>(nullptr), static_cast<int32_t*>(nullptr));
        } else if (index.dtype() == DType::Int64) {
            dispatch_gather(static_cast<double*>(nullptr), static_cast<int64_t*>(nullptr));
        } else {
            throw std::runtime_error("Unsupported index DType for pygo::gather. Expected Int32 or Int64.");
        }
    } else if (input.dtype() == DType::Int32) {
        if (index.dtype() == DType::Int32) {
            dispatch_gather(static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr));
        } else if (index.dtype() == DType::Int64) {
            dispatch_gather(static_cast<int32_t*>(nullptr), static_cast<int64_t*>(nullptr));
        } else {
            throw std::runtime_error("Unsupported index DType for pygo::gather. Expected Int32 or Int64.");
        }
    } else if (input.dtype() == DType::Int64) {
        if (index.dtype() == DType::Int32) {
            dispatch_gather(static_cast<int64_t*>(nullptr), static_cast<int32_t*>(nullptr));
        } else if (index.dtype() == DType::Int64) {
            dispatch_gather(static_cast<int64_t*>(nullptr), static_cast<int64_t*>(nullptr));
        } else {
            throw std::runtime_error("Unsupported index DType for pygo::gather. Expected Int32 or Int64.");
        }
    } else {
        throw std::runtime_error("Unsupported input DType for pygo::gather.");
    }

    return output;
}

} // namespace pygo
#include "pygo/block_scan.h"
#include <stdexcept>
#include <vector>
#include <numeric>

namespace pygo {

namespace detail {

template <typename T>
void cumsum_1d_cpu(const T* src, T* dst, int64_t n) {
    if (n <= 0) return;
    T running_sum = 0;
    for (int64_t i = 0; i < n; ++i) {
        running_sum += src[i];
        dst[i] = running_sum;
    }
}

template <typename T>
void cumprod_1d_cpu(const T* src, T* dst, int64_t n) {
    if (n <= 0) return;
    T running_prod = 1;
    for (int64_t i = 0; i < n; ++i) {
        running_prod *= src[i];
        dst[i] = running_prod;
    }
}

} // namespace detail

Tensor cumsum(const Tensor& input, int64_t dim) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::cumsum is currently supported only on CPU.");
    }

    Tensor input_c = input.contiguous();
    int64_t ndim = input_c.ndim();

    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("pygo::cumsum: dimension out of range.");
    }

    Tensor out = empty(input_c.shape(), input_c.dtype(), Device::CPU);
    int64_t shape_dim = input_c.shape().dims[dim];
    int64_t numel = input_c.numel();

    if (numel == 0) return out;

    // For simplicity of multi-dimensional scan, if 1D or flattened:
    if (ndim == 1 || (ndim == 2 && dim == 0)) {
        if (input_c.dtype() == DataType::Float32) {
            detail::cumsum_1d_cpu<float>(input_c.data<float>(), out.mutable_data<float>(), numel);
        } else if (input_c.dtype() == DataType::Float64) {
            detail::cumsum_1d_cpu<double>(input_c.data<double>(), out.mutable_data<double>(), numel);
        } else if (input_c.dtype() == DataType::Int32) {
            detail::cumsum_1d_cpu<int32_t>(input_c.data<int32_t>(), out.mutable_data<int32_t>(), numel);
        } else if (input_c.dtype() == DataType::Int64) {
            detail::cumsum_1d_cpu<int64_t>(input_c.data<int64_t>(), out.mutable_data<int64_t>(), numel);
        } else {
            throw std::runtime_error("Unsupported DataType for pygo::cumsum.");
        }
        return out;
    }

    throw std::runtime_error("pygo::cumsum along arbitrary dimensions is not yet fully implemented.");
}

Tensor cumprod(const Tensor& input, int64_t dim) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::cumprod is currently supported only on CPU.");
    }

    Tensor input_c = input.contiguous();
    int64_t ndim = input_c.ndim();

    if (dim < 0) {
        dim += ndim;
    }
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("pygo::cumprod: dimension out of range.");
    }

    Tensor out = empty(input_c.shape(), input_c.dtype(), Device::CPU);
    int64_t numel = input_c.numel();

    if (numel == 0) return out;

    if (ndim == 1 || (ndim == 2 && dim == 0)) {
        if (input_c.dtype() == DataType::Float32) {
            detail::cumprod_1d_cpu<float>(input_c.data<float>(), out.mutable_data<float>(), numel);
        } else if (input_c.dtype() == DataType::Float64) {
            detail::cumprod_1d_cpu<double>(input_c.data<double>(), out.mutable_data<double>(), numel);
        } else if (input_c.dtype() == DataType::Int32) {
            detail::cumprod_1d_cpu<int32_t>(input_c.data<int32_t>(), out.mutable_data<int32_t>(), numel);
        } else if (input_c.dtype() == DataType::Int64) {
            detail::cumprod_1d_cpu<int64_t>(input_c.data<int64_t>(), out.mutable_data<int64_t>(), numel);
        } else {
            throw std::runtime_error("Unsupported DataType for pygo::cumprod.");
        }
        return out;
    }

    throw std::runtime_error("pygo::cumprod along arbitrary dimensions is not yet fully implemented.");
}

} // namespace pygo
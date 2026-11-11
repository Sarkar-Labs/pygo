#include "pygo/flatten.h"
#include "pygo/dtype.h"
#include <stdexcept>
#include <vector>
#include <cstring>

namespace pygo {

namespace detail {

template <typename T>
void copy_tensor_data(const T* src, T* dst, int64_t numel) {
    std::memcpy(dst, src, numel * sizeof(T));
}

} // namespace detail

Tensor flatten(const Tensor& input, int64_t start_dim, int64_t end_dim) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::flatten is currently supported only on CPU.");
    }

    int64_t ndim = static_cast<int64_t>(input.shape().ndim());
    if (ndim == 0) {
        if (start_dim == 0 && (end_dim == 0 || end_dim == -1)) {
            return input;
        }
        throw std::out_of_range("Dimension out of range for 0-dimensional tensor in pygo::flatten.");
    }

    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;

    if (start_dim < 0 || start_dim >= ndim) {
        throw std::out_of_range("start_dim out of range in pygo::flatten.");
    }
    if (end_dim < 0 || end_dim >= ndim) {
        throw std::out_of_range("end_dim out of range in pygo::flatten.");
    }
    if (start_dim > end_dim) {
        throw std::invalid_argument("start_dim cannot be greater than end_dim in pygo::flatten.");
    }

    const auto& old_shape = input.shape();
    std::vector<int64_t> new_shape;

    for (int64_t i = 0; i < start_dim; ++i) {
        new_shape.push_back(old_shape[i]);
    }

    int64_t flattened_dim = 1;
    for (int64_t i = start_dim; i <= end_dim; ++i) {
        flattened_dim *= old_shape[i];
    }
    new_shape.push_back(flattened_dim);

    for (int64_t i = end_dim + 1; i < ndim; ++i) {
        new_shape.push_back(old_shape[i]);
    }

    auto output = zeros(Shape(new_shape), input.dtype(), input.device());
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
        } else if (input.dtype() == DType::Bool) {
            detail::copy_tensor_data(input.data<bool>(), output.data<bool>(), numel);
        } else {
            throw std::runtime_error("Unsupported DType for pygo::flatten.");
        }
    }

    return output;
}

} // namespace pygo
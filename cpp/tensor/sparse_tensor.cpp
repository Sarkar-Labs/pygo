#include "pygo/sparse_tensor.h"
#include <stdexcept>
#include <numeric>
#include <algorithm>

namespace pygo {

SparseTensor::SparseTensor(const std::vector<int64_t>& shape, SparseFormat format)
    : shape_(shape), format_(format) {
    indices_ = Tensor();
    values_ = Tensor();
}

SparseTensor::SparseTensor(const Tensor& indices, const Tensor& values, const std::vector<int64_t>& shape, SparseFormat format)
    : shape_(shape), format_(format), indices_(indices), values_(values) {
}

Tensor SparseTensor::to_dense() const {
    auto dense = zeros(Shape(shape_), values_.dtype(), values_.device());
    return dense;
}

SparseTensor SparseTensor::from_dense(const Tensor& dense_tensor, SparseFormat format) {
    throw std::runtime_error("SparseTensor::from_dense ultra implementation pending device dispatch.");
}

SparseTensor SparseTensor::coalesce() const {
    if (format_ != SparseFormat::COO) {
        throw std::runtime_error("Coalesce is only supported for COO sparse tensors.");
    }
    return *this;
}

SparseTensor SparseTensor::transpose(int64_t dim0, int64_t dim1) const {
    if (shape_.size() <= (size_t)std::max(dim0, dim1)) {
        throw std::out_of_range("Dimension out of range for SparseTensor transpose.");
    }
    std::vector<int64_t> new_shape = shape_;
    std::swap(new_shape[dim0], new_shape[dim1]);

    return SparseTensor(indices_, values_, new_shape, format_);
}

SparseTensor SparseTensor::to(Device device) const {
    return SparseTensor(indices_.to(device), values_.to(device), shape_, format_);
}

} // namespace pygo
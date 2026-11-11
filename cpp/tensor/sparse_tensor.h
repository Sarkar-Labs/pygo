#pragma once

#include "pygo/types.h"
#include "pygo/device.h"
#include "pygo/tensor.h"
#include <vector>
#include <memory>

namespace pygo {

enum class SparseFormat {
    COO, // Coordinate list format
    CSR  // Compressed Sparse Row format
};

class SparseTensor {
public:
    SparseTensor() = default;
    SparseTensor(const std::vector<int64_t>& shape, SparseFormat format = SparseFormat::COO);
    SparseTensor(const Tensor& indices, const Tensor& values, const std::vector<int64_t>& shape, SparseFormat format = SparseFormat::COO);

    const std::vector<int64_t>& shape() const { return shape_; }
    SparseFormat format() const { return format_; }
    
    Tensor indices() const { return indices_; }
    Tensor values() const { return values_; }

    Tensor to_dense() const;
    static SparseTensor from_dense(const Tensor& dense_tensor, SparseFormat format = SparseFormat::COO);

    SparseTensor coalesce() const;

    SparseTensor transpose(int64_t dim0, int64_t dim1) const;
    SparseTensor to(Device device) const;

private:
    std::vector<int64_t> shape_;
    SparseFormat format_ = SparseFormat::COO;
    Tensor indices_;
    Tensor values_;
};

} // namespace pygo
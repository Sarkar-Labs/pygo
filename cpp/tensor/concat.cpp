#include "pygo/concat.h"
#include <stdexcept>
#include <cstring>
#include <numeric>
#include <algorithm>

namespace pygo {

Tensor concat(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        throw std::invalid_argument("pygo::concat requires at least one tensor.");
    }

    const auto& first_shape = tensors[0].shape();
    int64_t ndim = static_cast<int64_t>(first_shape.size());

    if (ndim == 0) {
        throw std::invalid_argument("pygo::concat cannot be performed on 0-dim tensors.");
    }

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("Dimension out of range in pygo::concat.");
    }

    DType dtype = tensors[0].dtype();
    Device device = tensors[0].device();

    int64_t total_dim_size = 0;
    for (size_t i = 0; i < tensors.size(); ++i) {
        if (tensors[i].device() != device) {
            throw std::invalid_argument("All tensors in pygo::concat must be on the same device.");
        }
        if (tensors[i].dtype() != dtype) {
            throw std::invalid_argument("All tensors in pygo::concat must have the same dtype.");
        }

        const auto& s = tensors[i].shape();
        if (static_cast<int64_t>(s.size()) != ndim) {
            throw std::invalid_argument("All tensors in pygo::concat must have the same number of dimensions.");
        }

        for (int64_t d = 0; d < ndim; ++d) {
            if (d != dim && s[d] != first_shape[d]) {
                throw std::invalid_argument("Tensor shapes must match along non-concatenated dimensions.");
            }
        }
        total_dim_size += s[dim];
    }

    std::vector<int64_t> out_shape = first_shape.dims;
    out_shape[dim] = total_dim_size;

    auto output = zeros(Shape(out_shape), dtype, device);

    if (!device.is_cpu()) {
        throw std::runtime_error("pygo::concat is currently supported only on CPU.");
    }

    int64_t outer_size = 1;
    for (int64_t d = 0; d < dim; ++d) {
        outer_size *= first_shape[d];
    }

    int64_t inner_size = 1;
    for (int64_t d = dim + 1; d < ndim; ++d) {
        inner_size *= first_shape[d];
    }

    size_t itemsize = dtype_size(output.dtype());
    uint8_t* dst_ptr = output.data<uint8_t>();

    #pragma omp parallel for schedule(static) if(outer_size > 1)
    for (int64_t o = 0; o < outer_size; ++o) {
        int64_t dst_offset = o * total_dim_size * inner_size;
        for (size_t t = 0; t < tensors.size(); ++t) {
            int64_t current_dim_size = tensors[t].shape()[dim];
            const uint8_t* src_ptr = tensors[t].data<uint8_t>();
            int64_t src_offset = o * current_dim_size * inner_size;

            std::memcpy(dst_ptr + (dst_offset * itemsize),
                        src_ptr + (src_offset * itemsize),
                        current_dim_size * inner_size * itemsize);

            dst_offset += current_dim_size * inner_size;
        }
    }

    return output;
}

} // namespace pygo
#include "pygo/pad.h"
#include <stdexcept>
#include <vector>
namespace pygo {

Tensor pad(const Tensor& input, const std::vector<int64_t>& padding, double value) {
    if (padding.size() % 2 != 0)
        throw std::invalid_argument("pygo::pad: padding must have an even number of entries ([left,right] pairs)");
    int64_t n_padded_dims = (int64_t)padding.size() / 2;
    int64_t ndim = input.ndim();
    if (n_padded_dims > ndim)
        throw std::invalid_argument("pygo::pad: more padding pairs than input dimensions");
    if (input.dtype() != DType::Float32)
        throw std::runtime_error("pygo::pad: only Float32 is implemented so far");

    std::vector<int64_t> out_dims = input.shape().dims;
    std::vector<int64_t> left_pad(ndim, 0);
    for (int64_t i = 0; i < n_padded_dims; ++i) {
        int64_t d = ndim - 1 - i;
        int64_t left = padding[2*i], right = padding[2*i + 1];
        out_dims[d] += left + right;
        left_pad[d] = left;
    }

    Tensor out = full(Shape(out_dims), (float)value, DType::Float32, input.device());
    Tensor src = input.contiguous();

    // General, correct (if not the fastest) fix: copy element-by-element
    // using each side's own strides, rather than a flat Storage::copy_from
    // — which operates on raw buffers and would ignore both the offset
    // introduced by slicing and the gaps introduced by padding in outer
    // dimensions.
    const float* sp = src.data<float>();
    float* dp = out.data<float>();
    const auto& in_dims = input.shape().dims;
    const auto& out_strides = out.impl_->strides.dims; // out is freshly allocated, so contiguous
    int64_t n = src.numel();
    for (int64_t lin = 0; lin < n; ++lin) {
        int64_t rem = lin;
        int64_t dst_off = 0;
        for (int64_t k = ndim - 1; k >= 0; --k) {
            int64_t extent = in_dims[k];
            int64_t coord = rem % extent;
            rem /= extent;
            dst_off += (coord + left_pad[k]) * out_strides[k];
        }
        dp[dst_off] = sp[lin];
    }
    return out;
}

} // namespace pygo

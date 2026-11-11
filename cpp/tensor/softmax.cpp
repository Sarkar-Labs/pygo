#include "pygo/softmax.h"
namespace pygo {
Tensor softmax(const Tensor& input, int64_t dim) {
    int64_t d = dim < 0 ? dim + input.ndim() : dim;
    Tensor mx = input.max(d, /*keepdim=*/true);
    Tensor shifted = input - mx.expand(input.shape().dims);
    Tensor ex = shifted.exp();
    Tensor denom = ex.sum(d, /*keepdim=*/true);
    return ex / denom.expand(input.shape().dims);
}
} // namespace pygo

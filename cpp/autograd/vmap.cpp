#include "pygo/vmap.h"
#include <stdexcept>

namespace pygo::autograd {

Tensor vmap(std::function<Tensor(Tensor)> fn, const Tensor& input) {
    if (input.ndim() < 1) throw std::invalid_argument("pygo::autograd::vmap: input must have at least 1 dimension (dim 0 is the batch)");
    int64_t B = input.shape().dims[0];
    TensorList outputs;
    outputs.reserve(B);
    for (int64_t i = 0; i < B; ++i) {
        outputs.push_back(fn(input.index({i})).unsqueeze(0));
    }
    return cat(outputs, 0);
}

} // namespace pygo::autograd

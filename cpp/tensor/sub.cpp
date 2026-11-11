#include "pygo/sub.h"
#include <stdexcept>
namespace pygo {
Tensor sub(const Tensor& input, const Tensor& other) { return input - other; }
Tensor sub(const Tensor& input, double scalar) { return input + (float)(-scalar); }
void sub_(Tensor& input, const Tensor& other) {
    if (!input.impl_->is_contiguous()) throw std::runtime_error("sub_ requires a contiguous target tensor");
    Tensor result = (input - other).contiguous();
    input.impl_->storage->copy_from(*result.impl_->storage);
}
} // namespace pygo

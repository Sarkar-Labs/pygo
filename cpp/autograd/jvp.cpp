#include "pygo/jvp.h"
namespace pygo::autograd {
Tensor jvp(std::function<Tensor(Tensor)> fn, const Tensor& input, const Tensor& tangent, float h) {
    Tensor plus  = fn(input + tangent * h);
    Tensor minus = fn(input - tangent * h);
    return (plus - minus) * (1.0f / (2.0f * h));
}
} // namespace pygo::autograd

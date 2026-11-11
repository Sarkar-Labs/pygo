#include "pygo/hvp.h"
#include "pygo/engine.h"
namespace pygo::autograd {

namespace {
Tensor analytic_grad(std::function<Tensor(Tensor)> fn, const Tensor& x) {
    Tensor xi = x.detach();
    xi.set_requires_grad(true);
    Tensor y = fn(xi);
    TensorList grads = Engine::get().backward({y}, {}, false, false, {xi});
    return grads.empty() ? zeros(x.shape()) : grads[0];
}
}

Tensor hvp(std::function<Tensor(Tensor)> fn, const Tensor& input, const Tensor& v, float h) {
    Tensor g_plus  = analytic_grad(fn, input + v * h);
    Tensor g_minus = analytic_grad(fn, input - v * h);
    return (g_plus - g_minus) * (1.0f / (2.0f * h));
}
} // namespace pygo::autograd

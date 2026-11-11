#include "pygo/vjp.h"
#include "pygo/engine.h"
namespace pygo::autograd {
Tensor vjp(std::function<Tensor(Tensor)> fn, const Tensor& input, const Tensor& cotangent) {
    Tensor x = input.detach();
    x.set_requires_grad(true);
    Tensor y = fn(x);
    TensorList grads = Engine::get().backward({y}, {cotangent}, /*retain_graph=*/false, false, {x});
    return grads.empty() ? Tensor() : grads[0];
}
} // namespace pygo::autograd

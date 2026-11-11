#include "pygo/jacobian.h"
#include "pygo/engine.h"
#include <stdexcept>

namespace pygo::autograd {

Tensor jacobian(std::function<Tensor(Tensor)> fn, const Tensor& input) {
    Tensor x = input.detach();
    x.set_requires_grad(true);
    Tensor y = fn(x); // run once to discover output shape; re-run per row below
                       // since each backward() call needs its own fresh graph
                       // (retain_graph=false by default keeps memory bounded)
    int64_t out_n = y.numel();
    int64_t in_n = input.numel();
    if (out_n == 0 || in_n == 0) throw std::invalid_argument("pygo::autograd::jacobian: empty input or output");

    Tensor J = empty(Shape{out_n, in_n});
    for (int64_t i = 0; i < out_n; ++i) {
        Tensor xi = input.detach();
        xi.set_requires_grad(true);
        Tensor yi = fn(xi);
        Tensor cotangent = zeros(yi.shape());
        cotangent.data<float>()[i] = 1.0f; // one-hot: picks out row i of the Jacobian
        TensorList grads = Engine::get().backward({yi}, {cotangent}, false, false, {xi});
        Tensor gi = grads.empty() || !grads[0].defined() ? zeros(Shape{in_n}) : grads[0].contiguous().reshape({in_n});
        for (int64_t j = 0; j < in_n; ++j) J.data<float>()[i*in_n + j] = gi.data<float>()[j];
    }
    return J;
}

} // namespace pygo::autograd

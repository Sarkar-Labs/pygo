#include "pygo/retain.h"
#include "pygo/node.h"
#include <stdexcept>

namespace pygo::autograd {

void retain_grad(Tensor& t) {
    if (t.is_leaf())
        return; // leaves already always get their .grad populated via AccumulateGrad; nothing to do
    if (!t.impl_->grad_fn)
        throw std::runtime_error("pygo::autograd::retain_grad: tensor has no grad_fn (not part of a graph)");
    auto node = std::static_pointer_cast<Node>(t.impl_->grad_fn);
    node->retained_outputs.push_back(t);
}

} // namespace pygo::autograd

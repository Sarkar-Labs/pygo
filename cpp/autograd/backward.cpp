#include "pygo/backward.h"
#include "pygo/engine.h"
namespace pygo::autograd {
void backward(const TensorList& roots, const TensorList& root_grads, bool retain_graph) {
    Engine::get().backward(roots, root_grads, retain_graph, false, {});
}
void backward(const Tensor& root, const Tensor& root_grad, bool retain_graph) {
    TensorList grads = root_grad.defined() ? TensorList{root_grad} : TensorList{};
    Engine::get().backward({root}, grads, retain_graph, false, {});
}
} // namespace pygo::autograd

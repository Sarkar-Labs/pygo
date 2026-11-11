#include "pygo/accumulate.h"
#include <mutex>
#include <unordered_map>

namespace pygo::autograd {

TensorList AccumulateGrad::apply(TensorList grads) {
    if (grads.empty()) throw std::runtime_error("pygo::autograd::AccumulateGrad: no gradient to accumulate");
    Tensor g = grads[0];
    // Reuses Tensor's own accumulation logic (tensor.cpp: allocates .grad
    // on first use, adds into it on subsequent calls) rather than
    // duplicating that logic here.
    if (target.grad().defined()) {
        target.set_grad(target.grad() + g);
    } else {
        target.set_grad(g.clone());
    }
    return {}; // leaf node: nothing further upstream to propagate to
}

namespace {
// Keyed by TensorImpl identity so the same leaf always gets the same
// accumulator, even if it feeds multiple downstream ops (see accumulate.h's
// header comment for why that matters — otherwise concurrent paths would
// each overwrite `.grad` instead of summing into it).
std::mutex g_cache_mu;
std::unordered_map<void*, std::shared_ptr<Node>> g_accumulate_cache;
} // namespace

std::shared_ptr<Node> make_accumulate_grad_node(Tensor& leaf) {
    std::lock_guard<std::mutex> lk(g_cache_mu);
    void* key = leaf.impl_.get();
    auto it = g_accumulate_cache.find(key);
    if (it != g_accumulate_cache.end()) return it->second;

    auto node = std::make_shared<AccumulateGrad>();
    node->name = "AccumulateGrad";
    node->is_leaf = true;
    node->target = leaf;
    g_accumulate_cache[key] = node;
    return node;
}

} // namespace pygo::autograd

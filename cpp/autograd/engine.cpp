#include "pygo/engine.h"
#include "pygo/topological_sort.h"
#include "pygo/accumulate.h"
#include "pygo/function.h" // for grad_mode_enabled()
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace pygo::autograd {

// ─── Grad mode (thread-local, matches PyTorch's per-thread grad-enabled state) ─
namespace {
thread_local bool g_grad_enabled = true;
}

bool GradMode::is_enabled() { return g_grad_enabled; }
void GradMode::set_enabled(bool b) { g_grad_enabled = b; }
bool grad_mode_enabled() { return GradMode::is_enabled(); }

Engine& Engine::get() {
    static Engine instance;
    return instance;
}

TensorList Engine::backward(
    const TensorList& roots,
    const TensorList& root_grads,
    bool retain_graph,
    bool create_graph,
    const TensorList& inputs)
{
    (void)create_graph; // NOTE: building a graph *of the backward pass itself*
                         // (needed for higher-order derivatives) would require
                         // every Node::apply_fn to be expressed in terms of
                         // differentiable Tensor ops that themselves get
                         // wrapped in Nodes — not implemented; a single
                         // backward pass runs correctly, a second backward
                         // through it does not. Accepted for API compatibility
                         // and so callers get a clear place to look when
                         // double-backward doesn't do what they expect.

    if (roots.empty()) throw std::invalid_argument("pygo::autograd::Engine::backward: no roots given");
    if (!root_grads.empty() && root_grads.size() != roots.size())
        throw std::invalid_argument("pygo::autograd::Engine::backward: root_grads size must match roots size");

    // Collect the starting Node for each root. A root might itself be a
    // leaf (e.g. calling .backward() on a tensor with no ops applied to
    // it) — route those straight through an AccumulateGrad instead of
    // needing a graph at all.
    std::vector<std::shared_ptr<Node>> root_nodes;
    root_nodes.reserve(roots.size());
    for (auto& r : roots) {
        if (!r.requires_grad())
            throw std::runtime_error("pygo::autograd::Engine::backward: a root does not require_grad");
        if (r.impl_->grad_fn) {
            root_nodes.push_back(std::static_pointer_cast<Node>(r.impl_->grad_fn));
        } else {
            root_nodes.push_back(std::static_pointer_cast<Node>(make_accumulate_grad_node(const_cast<Tensor&>(r))));
        }
    }

    std::vector<std::shared_ptr<Node>> order = topological_sort(root_nodes);

    // grad_map[node.get()] accumulates every incoming gradient contribution
    // for that node before it runs (a node may have multiple consumers,
    // each contributing a partial gradient that must be *summed*, not
    // overwritten — this is the actual multi-input/multi-consumer chain
    // rule the original single-hop Tensor::backward() couldn't do).
    std::unordered_map<Node*, TensorList> grad_map;
    for (size_t i = 0; i < roots.size(); ++i) {
        Tensor g = (!root_grads.empty() && root_grads[i].defined())
                       ? root_grads[i]
                       : (roots[i].numel() == 1 ? ones(roots[i].shape(), roots[i].dtype(), roots[i].device())
                                                 : throw std::invalid_argument(
                                                       "pygo::autograd::Engine::backward: root_grads must be given for non-scalar roots"));
        auto& slot = grad_map[root_nodes[i].get()];
        if (slot.empty()) slot.push_back(g);
        else slot[0] = slot[0] + g;
    }

    // If `inputs` was given, only return gradients for those tensors
    // (computed as we go) instead of writing into every leaf's `.grad` —
    // this is the `torch.autograd.grad(outputs, inputs=...)` mode.
    std::unordered_map<TensorImpl*, Tensor> captured;
    std::unordered_set<TensorImpl*> want;
    for (auto& t : inputs) want.insert(t.impl_.get());

    for (auto& node : order) {
        auto it = grad_map.find(node.get());
        if (it == grad_map.end()) continue; // no gradient ever reached this node (disconnected branch)
        TensorList incoming = it->second;

        // If `node` is an AccumulateGrad the caller asked about via
        // `inputs`, capture its FULLY-ACCUMULATED incoming gradient right
        // here — this is the one point where `incoming` is guaranteed to
        // already contain every contribution (topological order means
        // every predecessor has already pushed into grad_map[node] by the
        // time we reach it).
        // NOTE (bug fix): this used to be captured inside the per-edge
        // distribution loop below instead, reading a single edge's
        // `out_grads[i]` and overwriting `captured[...]` once per
        // incoming edge — so whichever edge into the same AccumulateGrad
        // happened to run last silently won, instead of the sum of all of
        // them. Confirmed wrong with a 3-edges-into-one-leaf case (x0^3
        // built from two chained Mul nodes): it returned 4.0 instead of
        // the correct 12.0. Fixed by reading the already-summed
        // `incoming` here instead.
        if (auto* acc = dynamic_cast<AccumulateGrad*>(node.get())) {
            if (want.count(acc->target.impl_.get()) && !incoming.empty())
                captured[acc->target.impl_.get()] = incoming[0];
        }

        // execute_node: run this node's backward function with its
        // accumulated incoming gradient(s).
        TensorList out_grads = node->apply(incoming);

        // retain_grad() support: if any tensor asked to keep this node's
        // incoming gradient as its own .grad (see retain.h), do that now,
        // before the node's saved state is potentially cleared below.
        if (!node->retained_outputs.empty() && !incoming.empty()) {
            for (auto& t : node->retained_outputs) {
                Tensor tt = t; // non-const copy to call set_grad on
                if (tt.grad().defined()) tt.set_grad(tt.grad() + incoming[0]);
                else tt.set_grad(incoming[0].clone());
            }
        }

        // Route each output gradient to the corresponding next_fns entry
        // (paired 1:1 via next_input_nrs — see the NOTE added to GradFn in
        // tensor.h). Sum into that node's pending slot rather than
        // overwrite, since multiple upstream nodes can feed the same
        // downstream node.
        size_t n = std::min(out_grads.size(), node->next_fns.size());
        for (size_t i = 0; i < n; ++i) {
            auto next_gradfn = node->next_fns[i].lock();
            if (!next_gradfn) continue;
            Node* next = static_cast<Node*>(next_gradfn.get());
            auto& slot = grad_map[next];
            if (slot.empty()) slot.push_back(out_grads[i]);
            else slot[0] = slot[0] + out_grads[i];
        }

        if (!retain_graph) {
            // Drop this node's saved tensors and callable now that it has
            // run — mirrors PyTorch's default (single-use graph) memory
            // behavior. The node object itself may still be reachable via
            // shared_ptrs elsewhere (e.g. cached AccumulateGrad nodes), so
            // this only clears its payload, not the node itself.
            node->saved_tensors.clear();
            node->apply_fn = nullptr;
        }
    }

    if (inputs.empty()) return {};
    TensorList result;
    result.reserve(inputs.size());
    for (auto& t : inputs) {
        auto it = captured.find(t.impl_.get());
        result.push_back(it != captured.end() ? it->second : Tensor());
    }
    return result;
}

} // namespace pygo::autograd

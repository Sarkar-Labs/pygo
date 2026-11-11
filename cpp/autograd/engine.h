#pragma once
#include <vector>
#include <memory>
#include <unordered_map>
#include <functional>
#include <queue>
#include <mutex>
#include "pygo/tensor.h"

namespace pygo::autograd {

// ─── Node ─────────────────────────────────────────────────────────────────────
struct Node {
    using ApplyFn = std::function<TensorList(TensorList)>;

    std::string         name;
    ApplyFn             apply;
    std::vector<std::weak_ptr<Node>> next_nodes;  // edges to input nodes
    std::vector<Tensor> saved_tensors;
    int                 output_nr   = 0;
    bool                is_leaf     = false;
    std::atomic<int>    input_count {0};          // for topo-sort in-degree
    mutable std::mutex  mu;

    // Save tensors for backward
    void save_for_backward(TensorList tensors) { saved_tensors = std::move(tensors); }
    TensorList saved_inputs() const            { return saved_tensors; }
};

// ─── Edge ─────────────────────────────────────────────────────────────────────
struct Edge {
    std::shared_ptr<Node> fn;
    int input_nr = 0;
};

// ─── Engine ──────────────────────────────────────────────────────────────────
class Engine {
public:
    static Engine& get();

    // Main backward entry
    void backward(
        const TensorList& roots,
        const TensorList& root_grads,
        bool              retain_graph  = false,
        bool              create_graph  = false,
        const TensorList& inputs        = {}   // for grad() API
    );

    // Compute full Jacobian
    Tensor jacobian(const Tensor& out, const Tensor& inp,
                    bool create_graph=false);

    // Hessian via double-backward
    Tensor hessian(const Tensor& scalar, const Tensor& inp);

    // vmap-like functional vectorised grad
    Tensor vmap_grad(std::function<Tensor(Tensor)> fn,
                     const Tensor& inp, int in_dims=0, int out_dims=0);

    // JVP (forward-mode)
    Tensor jvp(std::function<Tensor(Tensor)> fn,
               const Tensor& inp, const Tensor& tangent);

    // VJP (reverse-mode, alias of backward)
    Tensor vjp(std::function<Tensor(Tensor)> fn,
               const Tensor& inp, const Tensor& cotangent);

    // Thread count for parallel backward
    void set_num_threads(int n) { num_threads_ = n; }
    int  num_threads()   const  { return num_threads_; }

private:
    Engine() = default;

    using ReadyQueue = std::priority_queue<
        std::pair<int, std::shared_ptr<Node>>,  // (seq_nr, node)
        std::vector<std::pair<int, std::shared_ptr<Node>>>,
        std::greater<>>;

    // Build topological order via BFS from roots
    std::vector<std::shared_ptr<Node>> topo_sort(
        const std::vector<std::shared_ptr<Node>>& roots);

    // Execute one node
    TensorList execute_node(
        Node& node,
        TensorList& grads,
        bool retain_graph);

    // Accumulate gradient into the grad map
    void accumulate_grad(
        const std::shared_ptr<Node>& node,
        TensorList& new_grads,
        std::unordered_map<Node*, TensorList>& grad_map);

    int num_threads_ = 1;
};

// ─── Grad Mode Guards ─────────────────────────────────────────────────────────
struct GradMode {
    static bool is_enabled();
    static void set_enabled(bool b);
};

struct NoGradGuard {
    NoGradGuard()  { prev_ = GradMode::is_enabled(); GradMode::set_enabled(false); }
    ~NoGradGuard() { GradMode::set_enabled(prev_); }
private:
    bool prev_;
};

struct EnableGradGuard {
    EnableGradGuard()  { prev_ = GradMode::is_enabled(); GradMode::set_enabled(true); }
    ~EnableGradGuard() { GradMode::set_enabled(prev_); }
private:
    bool prev_;
};

// ─── Activation Checkpoint ───────────────────────────────────────────────────
// Recomputes forward during backward to save memory
Tensor checkpoint(
    std::function<Tensor(TensorList)> fn,
    TensorList inputs,
    bool use_reentrant = true);

// ─── Custom Function API ─────────────────────────────────────────────────────
template<typename Derived>
struct Function {
    static Tensor apply(TensorList inputs) {
        auto ctx = std::make_shared<Derived>();
        TensorList outputs = Derived::forward(ctx.get(), inputs);
        if (GradMode::is_enabled()) {
            for (auto& out : outputs) {
                auto node = std::make_shared<Node>();
                node->name  = typeid(Derived).name();
                node->apply = [ctx](TensorList grads) {
                    return Derived::backward(ctx.get(), grads);
                };
                node->save_for_backward(inputs);
                out.impl_->grad_fn   = node;
                out.impl_->is_leaf   = false;
            }
        }
        return outputs.size() == 1 ? outputs[0] : outputs[0]; // simplified
    }
};

} // namespace pygo::autograd

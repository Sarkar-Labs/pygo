#include "pygo/topological_sort.h"
#include <queue>
#include <unordered_map>
#include <unordered_set>

namespace pygo::autograd {

std::vector<std::shared_ptr<Node>> topological_sort(const std::vector<std::shared_ptr<Node>>& roots) {
    // Pass 1: discover every node reachable from `roots` via next_fns.
    std::unordered_map<Node*, std::shared_ptr<Node>> all_nodes;
    std::vector<Node*> stack;
    for (auto& r : roots) {
        if (r && all_nodes.emplace(r.get(), r).second) stack.push_back(r.get());
    }
    while (!stack.empty()) {
        Node* n = stack.back(); stack.pop_back();
        for (auto& weak_next : n->next_fns) {
            auto next_gradfn = weak_next.lock();
            if (!next_gradfn) continue; // upstream node already freed
            // Safe: every GradFn this codebase's autograd builds is a Node
            // (Function<Derived>::apply_multi and AccumulateGrad both
            // construct real autograd::Node instances — see function.h,
            // accumulate.h). A GradFn from outside this system reaching
            // here would be a logic error elsewhere, not something to
            // silently tolerate, so this cast is intentionally unchecked.
            Node* next = static_cast<Node*>(next_gradfn.get());
            if (all_nodes.emplace(next, std::static_pointer_cast<Node>(next_gradfn)).second)
                stack.push_back(next);
        }
    }

    // Pass 2: compute in-degree within the discovered subgraph (how many
    // discovered nodes point at each node).
    std::unordered_map<Node*, int> in_degree;
    for (auto& [ptr, node] : all_nodes) in_degree[ptr] = 0;
    for (auto& [ptr, node] : all_nodes) {
        for (auto& weak_next : node->next_fns) {
            auto next_gradfn = weak_next.lock();
            if (!next_gradfn) continue;
            in_degree[static_cast<Node*>(next_gradfn.get())]++;
        }
    }
    for (auto& [ptr, node] : all_nodes) node->input_count.store(in_degree[ptr]);

    // Pass 3: Kahn's algorithm. Any node with in-degree 0 can run first —
    // in practice that's exactly `roots` (a root is a final forward output
    // that nothing else in this graph points at), computed generically here
    // rather than assumed.
    std::queue<Node*> ready;
    for (auto& [ptr, node] : all_nodes) if (in_degree[ptr] == 0) ready.push(ptr);

    std::vector<std::shared_ptr<Node>> order;
    std::unordered_set<Node*> emitted;
    while (!ready.empty()) {
        Node* n = ready.front(); ready.pop();
        if (!emitted.insert(n).second) continue;
        order.push_back(all_nodes[n]);
        for (auto& weak_next : n->next_fns) {
            auto next_gradfn = weak_next.lock();
            if (!next_gradfn) continue;
            Node* next = static_cast<Node*>(next_gradfn.get());
            if (--in_degree[next] == 0) ready.push(next);
        }
    }
    return order;
}

} // namespace pygo::autograd

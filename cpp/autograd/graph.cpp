#include "pygo/graph.h"
#include <unordered_map>
namespace pygo::autograd {
std::vector<std::shared_ptr<Node>> collect_graph(const std::vector<std::shared_ptr<Node>>& roots) {
    std::unordered_map<Node*, std::shared_ptr<Node>> seen;
    std::vector<Node*> stack;
    for (auto& r : roots) if (r && seen.emplace(r.get(), r).second) stack.push_back(r.get());
    while (!stack.empty()) {
        Node* n = stack.back(); stack.pop_back();
        for (auto& w : n->next_fns) {
            auto next = w.lock();
            if (!next) continue;
            Node* np = static_cast<Node*>(next.get());
            if (seen.emplace(np, std::static_pointer_cast<Node>(next)).second) stack.push_back(np);
        }
    }
    std::vector<std::shared_ptr<Node>> out;
    out.reserve(seen.size());
    for (auto& [ptr, node] : seen) out.push_back(node);
    return out;
}
} // namespace pygo::autograd

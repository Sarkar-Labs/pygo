// IR graph for PyGo computation
#include "pygo/core.h"
#include <sstream>
#include <algorithm>
namespace pygo::ir {

std::string IRGraph::dump() const {
    std::ostringstream os;
    os << "graph {\n";
    for (auto& n : nodes_) {
        os << "  ";
        for (auto* o : n->outputs) os << "%" << o->name << " ";
        os << "= " << n->op << "(";
        for (size_t i=0;i<n->inputs.size();++i) {
            os << "%" << n->inputs[i]->name;
            if (i+1 < n->inputs.size()) os << ", ";
        }
        os << ")\n";
    }
    os << "}\n";
    return os.str();
}

void IRGraph::dead_code_elimination() {
    // Remove nodes with no use of their outputs — EXCEPT nodes producing a
    // marked graph output (see mark_output()'s NOTE for why: a graph's own
    // final output legitimately has zero in-graph uses, and a naive
    // uses-empty check would incorrectly delete it too).
    nodes_.erase(std::remove_if(nodes_.begin(), nodes_.end(), [this](const auto& n) {
        for (auto* o : n->outputs) {
            if (!o->uses.empty()) return false;
            if (outputs_.count(o)) return false;
        }
        return true;
    }), nodes_.end());
}

void IRGraph::constant_folding() {
    // Evaluate all-constant subgraphs at compile time
}

void IRGraph::operator_fusion() {
    // Fuse elementwise chains: ReLU(LayerNorm(x)) -> fused kernel
}

void IRGraph::memory_planning() {
    // Assign buffers using liveness intervals for reuse
}

IRValue* IRGraph::add_value(const std::string& name, IRType type) {
    auto v = std::make_unique<IRValue>();
    v->name = name; v->type = type;
    IRValue* ptr = v.get();
    values_[name] = std::move(v);
    return ptr;
}

IRNode* IRGraph::add_node(const std::string& op,
    std::vector<IRValue*> ins, std::vector<IRValue*> outs,
    const std::unordered_map<std::string,std::string>& attrs)
{
    auto n = std::make_unique<IRNode>();
    n->op = op; n->inputs = ins; n->outputs = outs; n->attrs = attrs;
    IRNode* ptr = n.get();
    for (auto* o : outs) o->def = ptr;
    for (auto* i : ins)  i->uses.push_back(ptr);
    nodes_.push_back(std::move(n));
    return ptr;
}

} // namespace pygo::ir

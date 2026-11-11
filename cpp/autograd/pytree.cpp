#include "pygo/pytree.h"
#include <stdexcept>

namespace pygo::autograd {

namespace {
void flatten_into(const PyTree& t, TensorList& out) {
    if (t.is_leaf) { out.push_back(t.leaf); return; }
    for (auto& c : t.children) flatten_into(c, out);
}
PyTree unflatten_from(const PyTree& structure, const TensorList& flat, size_t& idx) {
    if (structure.is_leaf) {
        if (idx >= flat.size()) throw std::invalid_argument("pygo::autograd::tree_unflatten: not enough leaves supplied");
        return PyTree::Leaf(flat[idx++]);
    }
    std::vector<PyTree> children;
    children.reserve(structure.children.size());
    for (auto& c : structure.children) children.push_back(unflatten_from(c, flat, idx));
    return PyTree::Node(std::move(children));
}
} // namespace

TensorList tree_flatten(const PyTree& tree) {
    TensorList out;
    flatten_into(tree, out);
    return out;
}

PyTree tree_unflatten(const PyTree& structure, const TensorList& flat) {
    size_t idx = 0;
    PyTree result = unflatten_from(structure, flat, idx);
    if (idx != flat.size())
        throw std::invalid_argument("pygo::autograd::tree_unflatten: more leaves supplied than the structure has");
    return result;
}

} // namespace pygo::autograd

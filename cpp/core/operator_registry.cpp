#include "pygo/operator_registry.h"
#include <stdexcept>

namespace pygo::core {

OperatorRegistry& OperatorRegistry::instance() {
    static OperatorRegistry r;
    return r;
}

void OperatorRegistry::register_op(OpSchema schema) {
    ops_[schema.name] = std::move(schema);
}

bool OperatorRegistry::has(const std::string& name) const {
    return ops_.find(name) != ops_.end();
}

const OpSchema& OperatorRegistry::get(const std::string& name) const {
    auto it = ops_.find(name);
    if (it == ops_.end())
        throw std::runtime_error("pygo::core::OperatorRegistry: no schema registered for op \"" + name + "\"");
    return it->second;
}

void register_builtin_ops() {
    auto& reg = OperatorRegistry::instance();
    reg.register_op({"add", 2, 1, true, "Elementwise addition with broadcasting."});
    reg.register_op({"sub", 2, 1, true, "Elementwise subtraction with broadcasting."});
    reg.register_op({"mul", 2, 1, true, "Elementwise multiplication with broadcasting."});
    reg.register_op({"div", 2, 1, true, "Elementwise division with broadcasting."});
    reg.register_op({"neg", 1, 1, true, "Elementwise negation."});
    reg.register_op({"exp", 1, 1, true, "Elementwise exponential."});
    reg.register_op({"log", 1, 1, true, "Elementwise natural log."});
    reg.register_op({"sqrt", 1, 1, true, "Elementwise square root."});
    reg.register_op({"pow", 1, 1, true, "Elementwise power (scalar exponent)."});
    reg.register_op({"relu", 1, 1, true, "Rectified linear unit."});
    reg.register_op({"sigmoid", 1, 1, true, "Logistic sigmoid."});
    reg.register_op({"tanh", 1, 1, true, "Hyperbolic tangent."});
    reg.register_op({"gelu", 1, 1, true, "Gaussian Error Linear Unit (tanh approximation)."});
    reg.register_op({"silu", 1, 1, true, "Sigmoid Linear Unit (x * sigmoid(x))."});
    reg.register_op({"softmax", 1, 1, true, "Softmax along a given dimension."});
    reg.register_op({"matmul", 2, 1, true, "2D matrix multiplication."});
    reg.register_op({"sum", 1, 1, true, "Reduction: sum along a dimension (or full reduce)."});
    reg.register_op({"mean", 1, 1, true, "Reduction: mean along a dimension (or full reduce)."});
    reg.register_op({"reshape", 1, 1, true, "Reshape to a new (compatible) shape."});
    reg.register_op({"transpose", 1, 1, true, "Swap two dimensions."});
    reg.register_op({"cat", -1, 1, true, "Concatenate a list of tensors along a dimension."});
}

} // namespace pygo::core

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace pygo::ir {

enum class IRType {
    Unknown,
    Bool,
    Int64,
    Float32,
    Float64,
    Tensor,
};

struct IRNode;

struct IRValue {
    std::string name;
    IRType type{IRType::Unknown};
    IRNode* def{nullptr};
    std::vector<IRNode*> uses;
};

struct IRNode {
    std::string op;
    std::vector<IRValue*> inputs;
    std::vector<IRValue*> outputs;
    std::unordered_map<std::string, std::string> attrs;
};

class IRGraph {
public:
    IRGraph() = default;
    IRGraph(const IRGraph&) = delete;
    IRGraph& operator=(const IRGraph&) = delete;
    IRGraph(IRGraph&&) noexcept = default;
    IRGraph& operator=(IRGraph&&) noexcept = default;
    ~IRGraph() = default;

    std::string dump() const;
    void dead_code_elimination();
    void constant_folding();
    void operator_fusion();
    void memory_planning();

    IRValue* add_value(const std::string& name, IRType type = IRType::Unknown);
    IRNode* add_node(
        const std::string& op,
        std::vector<IRValue*> inputs,
        std::vector<IRValue*> outputs,
        const std::unordered_map<std::string, std::string>& attrs = {});

    void mark_output(IRValue* value) {
        if (value) outputs_.insert(value);
    }

private:
    std::vector<std::unique_ptr<IRValue>> values_storage_;
    std::unordered_map<std::string, std::unique_ptr<IRValue>> values_;
    std::vector<std::unique_ptr<IRNode>> nodes_;
    std::unordered_set<IRValue*> outputs_;
};

} // namespace pygo::ir

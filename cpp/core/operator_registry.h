#pragma once
#include <string>
#include <unordered_map>
namespace pygo::core {
struct OpSchema {
 std::string name;
 int inputs{0};
 int outputs{0};
 bool differentiable{false};
 std::string description;
};
class OperatorRegistry {
public:
 static OperatorRegistry& instance();
 void register_op(OpSchema schema);
 bool has(const std::string& name) const;
 const OpSchema& get(const std::string& name) const;
private:
 std::unordered_map<std::string, OpSchema> ops_;
};
void register_builtin_ops();
}

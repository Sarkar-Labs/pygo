#pragma once
#include "pygo/core.h"
#include <memory>
#include <string>
#include <vector>
namespace pygo::ir {
class Pass {
public:
 virtual ~Pass() = default;
 virtual const std::string& name() const = 0;
 virtual void run(IRGraph& graph) = 0;
};
class PassManager {
public:
 void add(std::unique_ptr<Pass> pass) { if (pass) passes_.push_back(std::move(pass)); }
 void run(IRGraph& graph, bool verbose=false) const;
 size_t size() const noexcept { return passes_.size(); }
private:
 std::vector<std::unique_ptr<Pass>> passes_;
};
}

#pragma once
#include "pygo/tensor.h"
#include <functional>
#include <mutex>
#include <string>
#include <unordered_map>
namespace pygo::core {
using KernelFn = std::function<Tensor(const TensorList&)>;
class KernelRegistry {
public:
 static KernelRegistry& instance();
 void register_kernel(const std::string& op, DeviceKind dev, DType dt, KernelFn fn);
 bool has(const std::string& op, DeviceKind dev, DType dt) const;
 KernelFn find(const std::string& op, DeviceKind dev, DType dt) const;
private:
 static std::string make_key(const std::string& op, DeviceKind dev, DType dt);
 mutable std::mutex mu_;
 std::unordered_map<std::string, KernelFn> kernels_;
};
}

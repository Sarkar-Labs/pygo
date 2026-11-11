#pragma once
#include "pygo/kernel_registry.h"
namespace pygo::core {
void register_kernel(const std::string& op, DeviceKind dev, DType dt, KernelFn fn);
Tensor dispatch(const std::string& op, const TensorList& inputs);
void set_autocast_dtype(DType dt);
DType get_autocast_dtype();
}

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0810_ceil_add_log(const Tensor&,const Tensor&);
Tensor fused_0810_ceil_add_log_contiguous(const Tensor&,const Tensor&);
Tensor fused_0810_ceil_add_log_cpu(const Tensor&,const Tensor&);
Tensor fused_0810_ceil_add_log_clone(const Tensor&,const Tensor&);
Tensor fused_0810_ceil_add_log_detached(const Tensor&,const Tensor&);
bool fused_0810_ceil_add_log_same_shape(const Tensor&,const Tensor&);
int64_t fused_0810_ceil_add_log_ndim(const Tensor&);
int64_t fused_0810_ceil_add_log_numel(const Tensor&);
DType fused_0810_ceil_add_log_dtype(const Tensor&);
Device fused_0810_ceil_add_log_device(const Tensor&);
std::string fused_0810_ceil_add_log_shape(const Tensor&);
const char* fused_0810_ceil_add_log_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

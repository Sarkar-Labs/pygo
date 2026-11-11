#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0285_tanh_add_gelu(const Tensor&,const Tensor&);
Tensor fused_0285_tanh_add_gelu_contiguous(const Tensor&,const Tensor&);
Tensor fused_0285_tanh_add_gelu_cpu(const Tensor&,const Tensor&);
Tensor fused_0285_tanh_add_gelu_clone(const Tensor&,const Tensor&);
Tensor fused_0285_tanh_add_gelu_detached(const Tensor&,const Tensor&);
bool fused_0285_tanh_add_gelu_same_shape(const Tensor&,const Tensor&);
int64_t fused_0285_tanh_add_gelu_ndim(const Tensor&);
int64_t fused_0285_tanh_add_gelu_numel(const Tensor&);
DType fused_0285_tanh_add_gelu_dtype(const Tensor&);
Device fused_0285_tanh_add_gelu_device(const Tensor&);
std::string fused_0285_tanh_add_gelu_shape(const Tensor&);
const char* fused_0285_tanh_add_gelu_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

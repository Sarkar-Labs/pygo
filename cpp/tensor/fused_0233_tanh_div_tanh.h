#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0233_tanh_div_tanh(const Tensor&,const Tensor&);
Tensor fused_0233_tanh_div_tanh_contiguous(const Tensor&,const Tensor&);
Tensor fused_0233_tanh_div_tanh_cpu(const Tensor&,const Tensor&);
Tensor fused_0233_tanh_div_tanh_clone(const Tensor&,const Tensor&);
Tensor fused_0233_tanh_div_tanh_detached(const Tensor&,const Tensor&);
bool fused_0233_tanh_div_tanh_same_shape(const Tensor&,const Tensor&);
int64_t fused_0233_tanh_div_tanh_ndim(const Tensor&);
int64_t fused_0233_tanh_div_tanh_numel(const Tensor&);
DType fused_0233_tanh_div_tanh_dtype(const Tensor&);
Device fused_0233_tanh_div_tanh_device(const Tensor&);
std::string fused_0233_tanh_div_tanh_shape(const Tensor&);
const char* fused_0233_tanh_div_tanh_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0238_tanh_div_abs(const Tensor&,const Tensor&);
Tensor fused_0238_tanh_div_abs_contiguous(const Tensor&,const Tensor&);
Tensor fused_0238_tanh_div_abs_cpu(const Tensor&,const Tensor&);
Tensor fused_0238_tanh_div_abs_clone(const Tensor&,const Tensor&);
Tensor fused_0238_tanh_div_abs_detached(const Tensor&,const Tensor&);
bool fused_0238_tanh_div_abs_same_shape(const Tensor&,const Tensor&);
int64_t fused_0238_tanh_div_abs_ndim(const Tensor&);
int64_t fused_0238_tanh_div_abs_numel(const Tensor&);
DType fused_0238_tanh_div_abs_dtype(const Tensor&);
Device fused_0238_tanh_div_abs_device(const Tensor&);
std::string fused_0238_tanh_div_abs_shape(const Tensor&);
const char* fused_0238_tanh_div_abs_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0173_sigmoid_div_sqrt(const Tensor&,const Tensor&);
Tensor fused_0173_sigmoid_div_sqrt_contiguous(const Tensor&,const Tensor&);
Tensor fused_0173_sigmoid_div_sqrt_cpu(const Tensor&,const Tensor&);
Tensor fused_0173_sigmoid_div_sqrt_clone(const Tensor&,const Tensor&);
Tensor fused_0173_sigmoid_div_sqrt_detached(const Tensor&,const Tensor&);
bool fused_0173_sigmoid_div_sqrt_same_shape(const Tensor&,const Tensor&);
int64_t fused_0173_sigmoid_div_sqrt_ndim(const Tensor&);
int64_t fused_0173_sigmoid_div_sqrt_numel(const Tensor&);
DType fused_0173_sigmoid_div_sqrt_dtype(const Tensor&);
Device fused_0173_sigmoid_div_sqrt_device(const Tensor&);
std::string fused_0173_sigmoid_div_sqrt_shape(const Tensor&);
const char* fused_0173_sigmoid_div_sqrt_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

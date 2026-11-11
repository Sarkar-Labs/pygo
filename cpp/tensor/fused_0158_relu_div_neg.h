#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0158_relu_div_neg(const Tensor&,const Tensor&);
Tensor fused_0158_relu_div_neg_contiguous(const Tensor&,const Tensor&);
Tensor fused_0158_relu_div_neg_cpu(const Tensor&,const Tensor&);
Tensor fused_0158_relu_div_neg_clone(const Tensor&,const Tensor&);
Tensor fused_0158_relu_div_neg_detached(const Tensor&,const Tensor&);
bool fused_0158_relu_div_neg_same_shape(const Tensor&,const Tensor&);
int64_t fused_0158_relu_div_neg_ndim(const Tensor&);
int64_t fused_0158_relu_div_neg_numel(const Tensor&);
DType fused_0158_relu_div_neg_dtype(const Tensor&);
Device fused_0158_relu_div_neg_device(const Tensor&);
std::string fused_0158_relu_div_neg_shape(const Tensor&);
const char* fused_0158_relu_div_neg_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

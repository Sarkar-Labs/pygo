#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0350_abs_add_sign(const Tensor&,const Tensor&);
Tensor fused_0350_abs_add_sign_contiguous(const Tensor&,const Tensor&);
Tensor fused_0350_abs_add_sign_cpu(const Tensor&,const Tensor&);
Tensor fused_0350_abs_add_sign_clone(const Tensor&,const Tensor&);
Tensor fused_0350_abs_add_sign_detached(const Tensor&,const Tensor&);
bool fused_0350_abs_add_sign_same_shape(const Tensor&,const Tensor&);
int64_t fused_0350_abs_add_sign_ndim(const Tensor&);
int64_t fused_0350_abs_add_sign_numel(const Tensor&);
DType fused_0350_abs_add_sign_dtype(const Tensor&);
Device fused_0350_abs_add_sign_device(const Tensor&);
std::string fused_0350_abs_add_sign_shape(const Tensor&);
const char* fused_0350_abs_add_sign_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

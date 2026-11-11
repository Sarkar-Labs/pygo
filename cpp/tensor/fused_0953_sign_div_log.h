#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0953_sign_div_log(const Tensor&,const Tensor&);
Tensor fused_0953_sign_div_log_contiguous(const Tensor&,const Tensor&);
Tensor fused_0953_sign_div_log_cpu(const Tensor&,const Tensor&);
Tensor fused_0953_sign_div_log_clone(const Tensor&,const Tensor&);
Tensor fused_0953_sign_div_log_detached(const Tensor&,const Tensor&);
bool fused_0953_sign_div_log_same_shape(const Tensor&,const Tensor&);
int64_t fused_0953_sign_div_log_ndim(const Tensor&);
int64_t fused_0953_sign_div_log_numel(const Tensor&);
DType fused_0953_sign_div_log_dtype(const Tensor&);
Device fused_0953_sign_div_log_device(const Tensor&);
std::string fused_0953_sign_div_log_shape(const Tensor&);
const char* fused_0953_sign_div_log_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

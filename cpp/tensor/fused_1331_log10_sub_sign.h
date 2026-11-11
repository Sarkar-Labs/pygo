#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1331_log10_sub_sign(const Tensor&,const Tensor&);
Tensor fused_1331_log10_sub_sign_contiguous(const Tensor&,const Tensor&);
Tensor fused_1331_log10_sub_sign_cpu(const Tensor&,const Tensor&);
Tensor fused_1331_log10_sub_sign_clone(const Tensor&,const Tensor&);
Tensor fused_1331_log10_sub_sign_detached(const Tensor&,const Tensor&);
bool fused_1331_log10_sub_sign_same_shape(const Tensor&,const Tensor&);
int64_t fused_1331_log10_sub_sign_ndim(const Tensor&);
int64_t fused_1331_log10_sub_sign_numel(const Tensor&);
DType fused_1331_log10_sub_sign_dtype(const Tensor&);
Device fused_1331_log10_sub_sign_device(const Tensor&);
std::string fused_1331_log10_sub_sign_shape(const Tensor&);
const char* fused_1331_log10_sub_sign_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

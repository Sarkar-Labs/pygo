#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0377_sqrt_mul_abs(const Tensor&,const Tensor&);
Tensor fused_0377_sqrt_mul_abs_contiguous(const Tensor&,const Tensor&);
Tensor fused_0377_sqrt_mul_abs_cpu(const Tensor&,const Tensor&);
Tensor fused_0377_sqrt_mul_abs_clone(const Tensor&,const Tensor&);
Tensor fused_0377_sqrt_mul_abs_detached(const Tensor&,const Tensor&);
bool fused_0377_sqrt_mul_abs_same_shape(const Tensor&,const Tensor&);
int64_t fused_0377_sqrt_mul_abs_ndim(const Tensor&);
int64_t fused_0377_sqrt_mul_abs_numel(const Tensor&);
DType fused_0377_sqrt_mul_abs_dtype(const Tensor&);
Device fused_0377_sqrt_mul_abs_device(const Tensor&);
std::string fused_0377_sqrt_mul_abs_shape(const Tensor&);
const char* fused_0377_sqrt_mul_abs_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

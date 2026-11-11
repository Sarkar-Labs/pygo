#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1342_log10_mul_silu(const Tensor&,const Tensor&);
Tensor fused_1342_log10_mul_silu_contiguous(const Tensor&,const Tensor&);
Tensor fused_1342_log10_mul_silu_cpu(const Tensor&,const Tensor&);
Tensor fused_1342_log10_mul_silu_clone(const Tensor&,const Tensor&);
Tensor fused_1342_log10_mul_silu_detached(const Tensor&,const Tensor&);
bool fused_1342_log10_mul_silu_same_shape(const Tensor&,const Tensor&);
int64_t fused_1342_log10_mul_silu_ndim(const Tensor&);
int64_t fused_1342_log10_mul_silu_numel(const Tensor&);
DType fused_1342_log10_mul_silu_dtype(const Tensor&);
Device fused_1342_log10_mul_silu_device(const Tensor&);
std::string fused_1342_log10_mul_silu_shape(const Tensor&);
const char* fused_1342_log10_mul_silu_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

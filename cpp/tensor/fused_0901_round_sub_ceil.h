#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0901_round_sub_ceil(const Tensor&,const Tensor&);
Tensor fused_0901_round_sub_ceil_contiguous(const Tensor&,const Tensor&);
Tensor fused_0901_round_sub_ceil_cpu(const Tensor&,const Tensor&);
Tensor fused_0901_round_sub_ceil_clone(const Tensor&,const Tensor&);
Tensor fused_0901_round_sub_ceil_detached(const Tensor&,const Tensor&);
bool fused_0901_round_sub_ceil_same_shape(const Tensor&,const Tensor&);
int64_t fused_0901_round_sub_ceil_ndim(const Tensor&);
int64_t fused_0901_round_sub_ceil_numel(const Tensor&);
DType fused_0901_round_sub_ceil_dtype(const Tensor&);
Device fused_0901_round_sub_ceil_device(const Tensor&);
std::string fused_0901_round_sub_ceil_shape(const Tensor&);
const char* fused_0901_round_sub_ceil_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

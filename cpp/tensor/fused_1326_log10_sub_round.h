#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1326_log10_sub_round(const Tensor&,const Tensor&);
Tensor fused_1326_log10_sub_round_contiguous(const Tensor&,const Tensor&);
Tensor fused_1326_log10_sub_round_cpu(const Tensor&,const Tensor&);
Tensor fused_1326_log10_sub_round_clone(const Tensor&,const Tensor&);
Tensor fused_1326_log10_sub_round_detached(const Tensor&,const Tensor&);
bool fused_1326_log10_sub_round_same_shape(const Tensor&,const Tensor&);
int64_t fused_1326_log10_sub_round_ndim(const Tensor&);
int64_t fused_1326_log10_sub_round_numel(const Tensor&);
DType fused_1326_log10_sub_round_dtype(const Tensor&);
Device fused_1326_log10_sub_round_device(const Tensor&);
std::string fused_1326_log10_sub_round_shape(const Tensor&);
const char* fused_1326_log10_sub_round_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

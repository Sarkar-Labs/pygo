#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1305_log10_add_sin(const Tensor&,const Tensor&);
Tensor fused_1305_log10_add_sin_contiguous(const Tensor&,const Tensor&);
Tensor fused_1305_log10_add_sin_cpu(const Tensor&,const Tensor&);
Tensor fused_1305_log10_add_sin_clone(const Tensor&,const Tensor&);
Tensor fused_1305_log10_add_sin_detached(const Tensor&,const Tensor&);
bool fused_1305_log10_add_sin_same_shape(const Tensor&,const Tensor&);
int64_t fused_1305_log10_add_sin_ndim(const Tensor&);
int64_t fused_1305_log10_add_sin_numel(const Tensor&);
DType fused_1305_log10_add_sin_dtype(const Tensor&);
Device fused_1305_log10_add_sin_device(const Tensor&);
std::string fused_1305_log10_add_sin_shape(const Tensor&);
const char* fused_1305_log10_add_sin_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

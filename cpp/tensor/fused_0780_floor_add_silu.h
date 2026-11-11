#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0780_floor_add_silu(const Tensor&,const Tensor&);
Tensor fused_0780_floor_add_silu_contiguous(const Tensor&,const Tensor&);
Tensor fused_0780_floor_add_silu_cpu(const Tensor&,const Tensor&);
Tensor fused_0780_floor_add_silu_clone(const Tensor&,const Tensor&);
Tensor fused_0780_floor_add_silu_detached(const Tensor&,const Tensor&);
bool fused_0780_floor_add_silu_same_shape(const Tensor&,const Tensor&);
int64_t fused_0780_floor_add_silu_ndim(const Tensor&);
int64_t fused_0780_floor_add_silu_numel(const Tensor&);
DType fused_0780_floor_add_silu_dtype(const Tensor&);
Device fused_0780_floor_add_silu_device(const Tensor&);
std::string fused_0780_floor_add_silu_shape(const Tensor&);
const char* fused_0780_floor_add_silu_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

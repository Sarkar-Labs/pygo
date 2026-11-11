#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1079_silu_matmul_abs(const Tensor&,const Tensor&);
Tensor fused_1079_silu_matmul_abs_contiguous(const Tensor&,const Tensor&);
Tensor fused_1079_silu_matmul_abs_cpu(const Tensor&,const Tensor&);
Tensor fused_1079_silu_matmul_abs_clone(const Tensor&,const Tensor&);
Tensor fused_1079_silu_matmul_abs_detached(const Tensor&,const Tensor&);
bool fused_1079_silu_matmul_abs_same_shape(const Tensor&,const Tensor&);
int64_t fused_1079_silu_matmul_abs_ndim(const Tensor&);
int64_t fused_1079_silu_matmul_abs_numel(const Tensor&);
DType fused_1079_silu_matmul_abs_dtype(const Tensor&);
Device fused_1079_silu_matmul_abs_device(const Tensor&);
std::string fused_1079_silu_matmul_abs_shape(const Tensor&);
const char* fused_1079_silu_matmul_abs_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

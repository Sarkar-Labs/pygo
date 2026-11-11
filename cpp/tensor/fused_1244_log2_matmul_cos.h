#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1244_log2_matmul_cos(const Tensor&,const Tensor&);
Tensor fused_1244_log2_matmul_cos_contiguous(const Tensor&,const Tensor&);
Tensor fused_1244_log2_matmul_cos_cpu(const Tensor&,const Tensor&);
Tensor fused_1244_log2_matmul_cos_clone(const Tensor&,const Tensor&);
Tensor fused_1244_log2_matmul_cos_detached(const Tensor&,const Tensor&);
bool fused_1244_log2_matmul_cos_same_shape(const Tensor&,const Tensor&);
int64_t fused_1244_log2_matmul_cos_ndim(const Tensor&);
int64_t fused_1244_log2_matmul_cos_numel(const Tensor&);
DType fused_1244_log2_matmul_cos_dtype(const Tensor&);
Device fused_1244_log2_matmul_cos_device(const Tensor&);
std::string fused_1244_log2_matmul_cos_shape(const Tensor&);
const char* fused_1244_log2_matmul_cos_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

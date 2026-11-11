#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0249_tanh_matmul_exp(const Tensor&,const Tensor&);
Tensor fused_0249_tanh_matmul_exp_contiguous(const Tensor&,const Tensor&);
Tensor fused_0249_tanh_matmul_exp_cpu(const Tensor&,const Tensor&);
Tensor fused_0249_tanh_matmul_exp_clone(const Tensor&,const Tensor&);
Tensor fused_0249_tanh_matmul_exp_detached(const Tensor&,const Tensor&);
bool fused_0249_tanh_matmul_exp_same_shape(const Tensor&,const Tensor&);
int64_t fused_0249_tanh_matmul_exp_ndim(const Tensor&);
int64_t fused_0249_tanh_matmul_exp_numel(const Tensor&);
DType fused_0249_tanh_matmul_exp_dtype(const Tensor&);
Device fused_0249_tanh_matmul_exp_device(const Tensor&);
std::string fused_0249_tanh_matmul_exp_shape(const Tensor&);
const char* fused_0249_tanh_matmul_exp_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

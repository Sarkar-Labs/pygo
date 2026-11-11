#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0724_floor_matmul_tanh(const Tensor&,const Tensor&);
Tensor fused_0724_floor_matmul_tanh_contiguous(const Tensor&,const Tensor&);
Tensor fused_0724_floor_matmul_tanh_cpu(const Tensor&,const Tensor&);
Tensor fused_0724_floor_matmul_tanh_clone(const Tensor&,const Tensor&);
Tensor fused_0724_floor_matmul_tanh_detached(const Tensor&,const Tensor&);
bool fused_0724_floor_matmul_tanh_same_shape(const Tensor&,const Tensor&);
int64_t fused_0724_floor_matmul_tanh_ndim(const Tensor&);
int64_t fused_0724_floor_matmul_tanh_numel(const Tensor&);
DType fused_0724_floor_matmul_tanh_dtype(const Tensor&);
Device fused_0724_floor_matmul_tanh_device(const Tensor&);
std::string fused_0724_floor_matmul_tanh_shape(const Tensor&);
const char* fused_0724_floor_matmul_tanh_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

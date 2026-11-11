#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0209_sigmoid_matmul_round(const Tensor&,const Tensor&);
Tensor fused_0209_sigmoid_matmul_round_contiguous(const Tensor&,const Tensor&);
Tensor fused_0209_sigmoid_matmul_round_cpu(const Tensor&,const Tensor&);
Tensor fused_0209_sigmoid_matmul_round_clone(const Tensor&,const Tensor&);
Tensor fused_0209_sigmoid_matmul_round_detached(const Tensor&,const Tensor&);
bool fused_0209_sigmoid_matmul_round_same_shape(const Tensor&,const Tensor&);
int64_t fused_0209_sigmoid_matmul_round_ndim(const Tensor&);
int64_t fused_0209_sigmoid_matmul_round_numel(const Tensor&);
DType fused_0209_sigmoid_matmul_round_dtype(const Tensor&);
Device fused_0209_sigmoid_matmul_round_device(const Tensor&);
std::string fused_0209_sigmoid_matmul_round_shape(const Tensor&);
const char* fused_0209_sigmoid_matmul_round_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0759_floor_matmul_floor(const Tensor&,const Tensor&);
Tensor fused_0759_floor_matmul_floor_contiguous(const Tensor&,const Tensor&);
Tensor fused_0759_floor_matmul_floor_cpu(const Tensor&,const Tensor&);
Tensor fused_0759_floor_matmul_floor_clone(const Tensor&,const Tensor&);
Tensor fused_0759_floor_matmul_floor_detached(const Tensor&,const Tensor&);
bool fused_0759_floor_matmul_floor_same_shape(const Tensor&,const Tensor&);
int64_t fused_0759_floor_matmul_floor_ndim(const Tensor&);
int64_t fused_0759_floor_matmul_floor_numel(const Tensor&);
DType fused_0759_floor_matmul_floor_dtype(const Tensor&);
Device fused_0759_floor_matmul_floor_device(const Tensor&);
std::string fused_0759_floor_matmul_floor_shape(const Tensor&);
const char* fused_0759_floor_matmul_floor_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

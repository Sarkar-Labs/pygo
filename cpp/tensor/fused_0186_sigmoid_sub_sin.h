#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0186_sigmoid_sub_sin(const Tensor&,const Tensor&);
Tensor fused_0186_sigmoid_sub_sin_contiguous(const Tensor&,const Tensor&);
Tensor fused_0186_sigmoid_sub_sin_cpu(const Tensor&,const Tensor&);
Tensor fused_0186_sigmoid_sub_sin_clone(const Tensor&,const Tensor&);
Tensor fused_0186_sigmoid_sub_sin_detached(const Tensor&,const Tensor&);
bool fused_0186_sigmoid_sub_sin_same_shape(const Tensor&,const Tensor&);
int64_t fused_0186_sigmoid_sub_sin_ndim(const Tensor&);
int64_t fused_0186_sigmoid_sub_sin_numel(const Tensor&);
DType fused_0186_sigmoid_sub_sin_dtype(const Tensor&);
Device fused_0186_sigmoid_sub_sin_device(const Tensor&);
std::string fused_0186_sigmoid_sub_sin_shape(const Tensor&);
const char* fused_0186_sigmoid_sub_sin_name();
}

// Contract: preserves the Tensor backend for allocation, dtype promotion, autograd, and device semantics.

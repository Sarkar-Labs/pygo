#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0052_ceil_mul(const Tensor&,const Tensor&);
Tensor fused_0052_ceil_mul_contiguous(const Tensor&,const Tensor&);
Tensor fused_0052_ceil_mul_cpu(const Tensor&,const Tensor&);
Tensor fused_0052_ceil_mul_clone(const Tensor&,const Tensor&);
Tensor fused_0052_ceil_mul_detached(const Tensor&,const Tensor&);
bool fused_0052_ceil_mul_same_shape(const Tensor&,const Tensor&);
int64_t fused_0052_ceil_mul_numel(const Tensor&);
std::string fused_0052_ceil_mul_shape(const Tensor&);
const char* fused_0052_ceil_mul_name();
}

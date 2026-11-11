#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0022_sqrt_mul(const Tensor&,const Tensor&);
Tensor fused_0022_sqrt_mul_contiguous(const Tensor&,const Tensor&);
Tensor fused_0022_sqrt_mul_cpu(const Tensor&,const Tensor&);
Tensor fused_0022_sqrt_mul_clone(const Tensor&,const Tensor&);
Tensor fused_0022_sqrt_mul_detached(const Tensor&,const Tensor&);
bool fused_0022_sqrt_mul_same_shape(const Tensor&,const Tensor&);
int64_t fused_0022_sqrt_mul_numel(const Tensor&);
std::string fused_0022_sqrt_mul_shape(const Tensor&);
const char* fused_0022_sqrt_mul_name();
}

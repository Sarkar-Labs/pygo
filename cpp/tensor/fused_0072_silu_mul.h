#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0072_silu_mul(const Tensor&,const Tensor&);
Tensor fused_0072_silu_mul_contiguous(const Tensor&,const Tensor&);
Tensor fused_0072_silu_mul_cpu(const Tensor&,const Tensor&);
Tensor fused_0072_silu_mul_clone(const Tensor&,const Tensor&);
Tensor fused_0072_silu_mul_detached(const Tensor&,const Tensor&);
bool fused_0072_silu_mul_same_shape(const Tensor&,const Tensor&);
int64_t fused_0072_silu_mul_numel(const Tensor&);
std::string fused_0072_silu_mul_shape(const Tensor&);
const char* fused_0072_silu_mul_name();
}

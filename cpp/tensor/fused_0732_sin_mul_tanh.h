#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0732_sin_mul_tanh(const Tensor&,const Tensor&);
Tensor fused_0732_sin_mul_tanh_contiguous(const Tensor&,const Tensor&);
Tensor fused_0732_sin_mul_tanh_cpu(const Tensor&,const Tensor&);
Tensor fused_0732_sin_mul_tanh_clone(const Tensor&,const Tensor&);
Tensor fused_0732_sin_mul_tanh_detached(const Tensor&,const Tensor&);
bool fused_0732_sin_mul_tanh_same_shape(const Tensor&,const Tensor&);
int64_t fused_0732_sin_mul_tanh_numel(const Tensor&);
std::string fused_0732_sin_mul_tanh_shape(const Tensor&);
const char* fused_0732_sin_mul_tanh_name();
}

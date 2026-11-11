#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0157_relu_mul_gelu(const Tensor&,const Tensor&);
Tensor fused_0157_relu_mul_gelu_contiguous(const Tensor&,const Tensor&);
Tensor fused_0157_relu_mul_gelu_cpu(const Tensor&,const Tensor&);
Tensor fused_0157_relu_mul_gelu_clone(const Tensor&,const Tensor&);
Tensor fused_0157_relu_mul_gelu_detached(const Tensor&,const Tensor&);
bool fused_0157_relu_mul_gelu_same_shape(const Tensor&,const Tensor&);
int64_t fused_0157_relu_mul_gelu_numel(const Tensor&);
std::string fused_0157_relu_mul_gelu_shape(const Tensor&);
const char* fused_0157_relu_mul_gelu_name();
}

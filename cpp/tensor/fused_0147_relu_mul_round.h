#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0147_relu_mul_round(const Tensor&,const Tensor&);
Tensor fused_0147_relu_mul_round_contiguous(const Tensor&,const Tensor&);
Tensor fused_0147_relu_mul_round_cpu(const Tensor&,const Tensor&);
Tensor fused_0147_relu_mul_round_clone(const Tensor&,const Tensor&);
Tensor fused_0147_relu_mul_round_detached(const Tensor&,const Tensor&);
bool fused_0147_relu_mul_round_same_shape(const Tensor&,const Tensor&);
int64_t fused_0147_relu_mul_round_numel(const Tensor&);
std::string fused_0147_relu_mul_round_shape(const Tensor&);
const char* fused_0147_relu_mul_round_name();
}

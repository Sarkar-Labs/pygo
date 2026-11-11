#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0155_relu_add_gelu(const Tensor&,const Tensor&);
Tensor fused_0155_relu_add_gelu_contiguous(const Tensor&,const Tensor&);
Tensor fused_0155_relu_add_gelu_cpu(const Tensor&,const Tensor&);
Tensor fused_0155_relu_add_gelu_clone(const Tensor&,const Tensor&);
Tensor fused_0155_relu_add_gelu_detached(const Tensor&,const Tensor&);
bool fused_0155_relu_add_gelu_same_shape(const Tensor&,const Tensor&);
int64_t fused_0155_relu_add_gelu_numel(const Tensor&);
std::string fused_0155_relu_add_gelu_shape(const Tensor&);
const char* fused_0155_relu_add_gelu_name();
}

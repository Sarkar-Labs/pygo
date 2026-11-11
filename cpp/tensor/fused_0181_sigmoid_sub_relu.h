#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0181_sigmoid_sub_relu(const Tensor&,const Tensor&);
Tensor fused_0181_sigmoid_sub_relu_contiguous(const Tensor&,const Tensor&);
Tensor fused_0181_sigmoid_sub_relu_cpu(const Tensor&,const Tensor&);
Tensor fused_0181_sigmoid_sub_relu_clone(const Tensor&,const Tensor&);
Tensor fused_0181_sigmoid_sub_relu_detached(const Tensor&,const Tensor&);
bool fused_0181_sigmoid_sub_relu_same_shape(const Tensor&,const Tensor&);
int64_t fused_0181_sigmoid_sub_relu_numel(const Tensor&);
std::string fused_0181_sigmoid_sub_relu_shape(const Tensor&);
const char* fused_0181_sigmoid_sub_relu_name();
}

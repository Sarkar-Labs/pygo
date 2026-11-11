#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0004_relu_matmul(const Tensor&,const Tensor&);
Tensor fused_0004_relu_matmul_contiguous(const Tensor&,const Tensor&);
Tensor fused_0004_relu_matmul_cpu(const Tensor&,const Tensor&);
Tensor fused_0004_relu_matmul_clone(const Tensor&,const Tensor&);
Tensor fused_0004_relu_matmul_detached(const Tensor&,const Tensor&);
bool fused_0004_relu_matmul_same_shape(const Tensor&,const Tensor&);
int64_t fused_0004_relu_matmul_numel(const Tensor&);
std::string fused_0004_relu_matmul_shape(const Tensor&);
const char* fused_0004_relu_matmul_name();
}

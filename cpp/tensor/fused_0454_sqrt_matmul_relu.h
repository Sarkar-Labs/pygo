#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0454_sqrt_matmul_relu(const Tensor&,const Tensor&);
Tensor fused_0454_sqrt_matmul_relu_contiguous(const Tensor&,const Tensor&);
Tensor fused_0454_sqrt_matmul_relu_cpu(const Tensor&,const Tensor&);
Tensor fused_0454_sqrt_matmul_relu_clone(const Tensor&,const Tensor&);
Tensor fused_0454_sqrt_matmul_relu_detached(const Tensor&,const Tensor&);
bool fused_0454_sqrt_matmul_relu_same_shape(const Tensor&,const Tensor&);
int64_t fused_0454_sqrt_matmul_relu_numel(const Tensor&);
std::string fused_0454_sqrt_matmul_relu_shape(const Tensor&);
const char* fused_0454_sqrt_matmul_relu_name();
}

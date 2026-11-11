#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1239_sign_matmul_gelu(const Tensor&,const Tensor&);
Tensor fused_1239_sign_matmul_gelu_contiguous(const Tensor&,const Tensor&);
Tensor fused_1239_sign_matmul_gelu_cpu(const Tensor&,const Tensor&);
Tensor fused_1239_sign_matmul_gelu_clone(const Tensor&,const Tensor&);
Tensor fused_1239_sign_matmul_gelu_detached(const Tensor&,const Tensor&);
bool fused_1239_sign_matmul_gelu_same_shape(const Tensor&,const Tensor&);
int64_t fused_1239_sign_matmul_gelu_numel(const Tensor&);
std::string fused_1239_sign_matmul_gelu_shape(const Tensor&);
const char* fused_1239_sign_matmul_gelu_name();
}

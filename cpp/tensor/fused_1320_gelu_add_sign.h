#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1320_gelu_add_sign(const Tensor&,const Tensor&);
Tensor fused_1320_gelu_add_sign_contiguous(const Tensor&,const Tensor&);
Tensor fused_1320_gelu_add_sign_cpu(const Tensor&,const Tensor&);
Tensor fused_1320_gelu_add_sign_clone(const Tensor&,const Tensor&);
Tensor fused_1320_gelu_add_sign_detached(const Tensor&,const Tensor&);
bool fused_1320_gelu_add_sign_same_shape(const Tensor&,const Tensor&);
int64_t fused_1320_gelu_add_sign_numel(const Tensor&);
std::string fused_1320_gelu_add_sign_shape(const Tensor&);
const char* fused_1320_gelu_add_sign_name();
}

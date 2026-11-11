#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0065_gelu_add(const Tensor&,const Tensor&);
Tensor fused_0065_gelu_add_contiguous(const Tensor&,const Tensor&);
Tensor fused_0065_gelu_add_cpu(const Tensor&,const Tensor&);
Tensor fused_0065_gelu_add_clone(const Tensor&,const Tensor&);
Tensor fused_0065_gelu_add_detached(const Tensor&,const Tensor&);
bool fused_0065_gelu_add_same_shape(const Tensor&,const Tensor&);
int64_t fused_0065_gelu_add_numel(const Tensor&);
std::string fused_0065_gelu_add_shape(const Tensor&);
const char* fused_0065_gelu_add_name();
}

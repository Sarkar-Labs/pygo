#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0490_sqrt_add_cos(const Tensor&,const Tensor&);
Tensor fused_0490_sqrt_add_cos_contiguous(const Tensor&,const Tensor&);
Tensor fused_0490_sqrt_add_cos_cpu(const Tensor&,const Tensor&);
Tensor fused_0490_sqrt_add_cos_clone(const Tensor&,const Tensor&);
Tensor fused_0490_sqrt_add_cos_detached(const Tensor&,const Tensor&);
bool fused_0490_sqrt_add_cos_same_shape(const Tensor&,const Tensor&);
int64_t fused_0490_sqrt_add_cos_numel(const Tensor&);
std::string fused_0490_sqrt_add_cos_shape(const Tensor&);
const char* fused_0490_sqrt_add_cos_name();
}

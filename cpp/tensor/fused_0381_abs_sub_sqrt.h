#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0381_abs_sub_sqrt(const Tensor&,const Tensor&);
Tensor fused_0381_abs_sub_sqrt_contiguous(const Tensor&,const Tensor&);
Tensor fused_0381_abs_sub_sqrt_cpu(const Tensor&,const Tensor&);
Tensor fused_0381_abs_sub_sqrt_clone(const Tensor&,const Tensor&);
Tensor fused_0381_abs_sub_sqrt_detached(const Tensor&,const Tensor&);
bool fused_0381_abs_sub_sqrt_same_shape(const Tensor&,const Tensor&);
int64_t fused_0381_abs_sub_sqrt_numel(const Tensor&);
std::string fused_0381_abs_sub_sqrt_shape(const Tensor&);
const char* fused_0381_abs_sub_sqrt_name();
}

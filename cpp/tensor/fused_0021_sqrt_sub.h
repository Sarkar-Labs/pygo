#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0021_sqrt_sub(const Tensor&,const Tensor&);
Tensor fused_0021_sqrt_sub_contiguous(const Tensor&,const Tensor&);
Tensor fused_0021_sqrt_sub_cpu(const Tensor&,const Tensor&);
Tensor fused_0021_sqrt_sub_clone(const Tensor&,const Tensor&);
Tensor fused_0021_sqrt_sub_detached(const Tensor&,const Tensor&);
bool fused_0021_sqrt_sub_same_shape(const Tensor&,const Tensor&);
int64_t fused_0021_sqrt_sub_numel(const Tensor&);
std::string fused_0021_sqrt_sub_shape(const Tensor&);
const char* fused_0021_sqrt_sub_name();
}

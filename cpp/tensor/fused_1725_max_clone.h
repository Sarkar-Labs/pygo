#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1725_max_clone(const Tensor&,const Tensor&);
Tensor fused_1725_max_clone_contiguous(const Tensor&,const Tensor&);
Tensor fused_1725_max_clone_cpu(const Tensor&,const Tensor&);
Tensor fused_1725_max_clone_clone(const Tensor&,const Tensor&);
Tensor fused_1725_max_clone_detached(const Tensor&,const Tensor&);
bool fused_1725_max_clone_same_shape(const Tensor&,const Tensor&);
int64_t fused_1725_max_clone_numel(const Tensor&);
std::string fused_1725_max_clone_shape(const Tensor&);
const char* fused_1725_max_clone_name();
}

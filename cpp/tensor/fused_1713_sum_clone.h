#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1713_sum_clone(const Tensor&,const Tensor&);
Tensor fused_1713_sum_clone_contiguous(const Tensor&,const Tensor&);
Tensor fused_1713_sum_clone_cpu(const Tensor&,const Tensor&);
Tensor fused_1713_sum_clone_clone(const Tensor&,const Tensor&);
Tensor fused_1713_sum_clone_detached(const Tensor&,const Tensor&);
bool fused_1713_sum_clone_same_shape(const Tensor&,const Tensor&);
int64_t fused_1713_sum_clone_numel(const Tensor&);
std::string fused_1713_sum_clone_shape(const Tensor&);
const char* fused_1713_sum_clone_name();
}

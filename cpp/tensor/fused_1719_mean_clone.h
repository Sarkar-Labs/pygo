#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1719_mean_clone(const Tensor&,const Tensor&);
Tensor fused_1719_mean_clone_contiguous(const Tensor&,const Tensor&);
Tensor fused_1719_mean_clone_cpu(const Tensor&,const Tensor&);
Tensor fused_1719_mean_clone_clone(const Tensor&,const Tensor&);
Tensor fused_1719_mean_clone_detached(const Tensor&,const Tensor&);
bool fused_1719_mean_clone_same_shape(const Tensor&,const Tensor&);
int64_t fused_1719_mean_clone_numel(const Tensor&);
std::string fused_1719_mean_clone_shape(const Tensor&);
const char* fused_1719_mean_clone_name();
}

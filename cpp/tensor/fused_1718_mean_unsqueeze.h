#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1718_mean_unsqueeze(const Tensor&,const Tensor&);
Tensor fused_1718_mean_unsqueeze_contiguous(const Tensor&,const Tensor&);
Tensor fused_1718_mean_unsqueeze_cpu(const Tensor&,const Tensor&);
Tensor fused_1718_mean_unsqueeze_clone(const Tensor&,const Tensor&);
Tensor fused_1718_mean_unsqueeze_detached(const Tensor&,const Tensor&);
bool fused_1718_mean_unsqueeze_same_shape(const Tensor&,const Tensor&);
int64_t fused_1718_mean_unsqueeze_numel(const Tensor&);
std::string fused_1718_mean_unsqueeze_shape(const Tensor&);
const char* fused_1718_mean_unsqueeze_name();
}

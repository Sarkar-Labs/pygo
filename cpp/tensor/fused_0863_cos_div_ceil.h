#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0863_cos_div_ceil(const Tensor&,const Tensor&);
Tensor fused_0863_cos_div_ceil_contiguous(const Tensor&,const Tensor&);
Tensor fused_0863_cos_div_ceil_cpu(const Tensor&,const Tensor&);
Tensor fused_0863_cos_div_ceil_clone(const Tensor&,const Tensor&);
Tensor fused_0863_cos_div_ceil_detached(const Tensor&,const Tensor&);
bool fused_0863_cos_div_ceil_same_shape(const Tensor&,const Tensor&);
int64_t fused_0863_cos_div_ceil_numel(const Tensor&);
std::string fused_0863_cos_div_ceil_shape(const Tensor&);
const char* fused_0863_cos_div_ceil_name();
}

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1303_gelu_div_cos(const Tensor&,const Tensor&);
Tensor fused_1303_gelu_div_cos_contiguous(const Tensor&,const Tensor&);
Tensor fused_1303_gelu_div_cos_cpu(const Tensor&,const Tensor&);
Tensor fused_1303_gelu_div_cos_clone(const Tensor&,const Tensor&);
Tensor fused_1303_gelu_div_cos_detached(const Tensor&,const Tensor&);
bool fused_1303_gelu_div_cos_same_shape(const Tensor&,const Tensor&);
int64_t fused_1303_gelu_div_cos_numel(const Tensor&);
std::string fused_1303_gelu_div_cos_shape(const Tensor&);
const char* fused_1303_gelu_div_cos_name();
}

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0083_log2_div(const Tensor&,const Tensor&);
Tensor fused_0083_log2_div_contiguous(const Tensor&,const Tensor&);
Tensor fused_0083_log2_div_cpu(const Tensor&,const Tensor&);
Tensor fused_0083_log2_div_clone(const Tensor&,const Tensor&);
Tensor fused_0083_log2_div_detached(const Tensor&,const Tensor&);
bool fused_0083_log2_div_same_shape(const Tensor&,const Tensor&);
int64_t fused_0083_log2_div_numel(const Tensor&);
std::string fused_0083_log2_div_shape(const Tensor&);
const char* fused_0083_log2_div_name();
}

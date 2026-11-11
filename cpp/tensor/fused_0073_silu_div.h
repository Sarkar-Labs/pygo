#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0073_silu_div(const Tensor&,const Tensor&);
Tensor fused_0073_silu_div_contiguous(const Tensor&,const Tensor&);
Tensor fused_0073_silu_div_cpu(const Tensor&,const Tensor&);
Tensor fused_0073_silu_div_clone(const Tensor&,const Tensor&);
Tensor fused_0073_silu_div_detached(const Tensor&,const Tensor&);
bool fused_0073_silu_div_same_shape(const Tensor&,const Tensor&);
int64_t fused_0073_silu_div_numel(const Tensor&);
std::string fused_0073_silu_div_shape(const Tensor&);
const char* fused_0073_silu_div_name();
}

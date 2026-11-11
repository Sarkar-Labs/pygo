#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1423_silu_div_silu(const Tensor&,const Tensor&);
Tensor fused_1423_silu_div_silu_contiguous(const Tensor&,const Tensor&);
Tensor fused_1423_silu_div_silu_cpu(const Tensor&,const Tensor&);
Tensor fused_1423_silu_div_silu_clone(const Tensor&,const Tensor&);
Tensor fused_1423_silu_div_silu_detached(const Tensor&,const Tensor&);
bool fused_1423_silu_div_silu_same_shape(const Tensor&,const Tensor&);
int64_t fused_1423_silu_div_silu_numel(const Tensor&);
std::string fused_1423_silu_div_silu_shape(const Tensor&);
const char* fused_1423_silu_div_silu_name();
}

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1408_silu_div_round(const Tensor&,const Tensor&);
Tensor fused_1408_silu_div_round_contiguous(const Tensor&,const Tensor&);
Tensor fused_1408_silu_div_round_cpu(const Tensor&,const Tensor&);
Tensor fused_1408_silu_div_round_clone(const Tensor&,const Tensor&);
Tensor fused_1408_silu_div_round_detached(const Tensor&,const Tensor&);
bool fused_1408_silu_div_round_same_shape(const Tensor&,const Tensor&);
int64_t fused_1408_silu_div_round_numel(const Tensor&);
std::string fused_1408_silu_div_round_shape(const Tensor&);
const char* fused_1408_silu_div_round_name();
}

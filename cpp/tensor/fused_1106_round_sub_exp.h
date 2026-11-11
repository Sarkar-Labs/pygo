#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1106_round_sub_exp(const Tensor&,const Tensor&);
Tensor fused_1106_round_sub_exp_contiguous(const Tensor&,const Tensor&);
Tensor fused_1106_round_sub_exp_cpu(const Tensor&,const Tensor&);
Tensor fused_1106_round_sub_exp_clone(const Tensor&,const Tensor&);
Tensor fused_1106_round_sub_exp_detached(const Tensor&,const Tensor&);
bool fused_1106_round_sub_exp_same_shape(const Tensor&,const Tensor&);
int64_t fused_1106_round_sub_exp_numel(const Tensor&);
std::string fused_1106_round_sub_exp_shape(const Tensor&);
const char* fused_1106_round_sub_exp_name();
}

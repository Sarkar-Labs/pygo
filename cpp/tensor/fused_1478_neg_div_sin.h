#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1478_neg_div_sin(const Tensor&,const Tensor&);
Tensor fused_1478_neg_div_sin_contiguous(const Tensor&,const Tensor&);
Tensor fused_1478_neg_div_sin_cpu(const Tensor&,const Tensor&);
Tensor fused_1478_neg_div_sin_clone(const Tensor&,const Tensor&);
Tensor fused_1478_neg_div_sin_detached(const Tensor&,const Tensor&);
bool fused_1478_neg_div_sin_same_shape(const Tensor&,const Tensor&);
int64_t fused_1478_neg_div_sin_numel(const Tensor&);
std::string fused_1478_neg_div_sin_shape(const Tensor&);
const char* fused_1478_neg_div_sin_name();
}

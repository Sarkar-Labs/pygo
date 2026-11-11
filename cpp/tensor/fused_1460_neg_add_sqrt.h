#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1460_neg_add_sqrt(const Tensor&,const Tensor&);
Tensor fused_1460_neg_add_sqrt_contiguous(const Tensor&,const Tensor&);
Tensor fused_1460_neg_add_sqrt_cpu(const Tensor&,const Tensor&);
Tensor fused_1460_neg_add_sqrt_clone(const Tensor&,const Tensor&);
Tensor fused_1460_neg_add_sqrt_detached(const Tensor&,const Tensor&);
bool fused_1460_neg_add_sqrt_same_shape(const Tensor&,const Tensor&);
int64_t fused_1460_neg_add_sqrt_numel(const Tensor&);
std::string fused_1460_neg_add_sqrt_shape(const Tensor&);
const char* fused_1460_neg_add_sqrt_name();
}

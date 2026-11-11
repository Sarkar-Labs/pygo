#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1507_neg_mul_gelu(const Tensor&,const Tensor&);
Tensor fused_1507_neg_mul_gelu_contiguous(const Tensor&,const Tensor&);
Tensor fused_1507_neg_mul_gelu_cpu(const Tensor&,const Tensor&);
Tensor fused_1507_neg_mul_gelu_clone(const Tensor&,const Tensor&);
Tensor fused_1507_neg_mul_gelu_detached(const Tensor&,const Tensor&);
bool fused_1507_neg_mul_gelu_same_shape(const Tensor&,const Tensor&);
int64_t fused_1507_neg_mul_gelu_numel(const Tensor&);
std::string fused_1507_neg_mul_gelu_shape(const Tensor&);
const char* fused_1507_neg_mul_gelu_name();
}

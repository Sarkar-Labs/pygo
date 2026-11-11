#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1429_silu_matmul_neg(const Tensor&,const Tensor&);
Tensor fused_1429_silu_matmul_neg_contiguous(const Tensor&,const Tensor&);
Tensor fused_1429_silu_matmul_neg_cpu(const Tensor&,const Tensor&);
Tensor fused_1429_silu_matmul_neg_clone(const Tensor&,const Tensor&);
Tensor fused_1429_silu_matmul_neg_detached(const Tensor&,const Tensor&);
bool fused_1429_silu_matmul_neg_same_shape(const Tensor&,const Tensor&);
int64_t fused_1429_silu_matmul_neg_numel(const Tensor&);
std::string fused_1429_silu_matmul_neg_shape(const Tensor&);
const char* fused_1429_silu_matmul_neg_name();
}

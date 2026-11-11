#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0044_cos_matmul(const Tensor&,const Tensor&);
Tensor fused_0044_cos_matmul_contiguous(const Tensor&,const Tensor&);
Tensor fused_0044_cos_matmul_cpu(const Tensor&,const Tensor&);
Tensor fused_0044_cos_matmul_clone(const Tensor&,const Tensor&);
Tensor fused_0044_cos_matmul_detached(const Tensor&,const Tensor&);
bool fused_0044_cos_matmul_same_shape(const Tensor&,const Tensor&);
int64_t fused_0044_cos_matmul_numel(const Tensor&);
std::string fused_0044_cos_matmul_shape(const Tensor&);
const char* fused_0044_cos_matmul_name();
}

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1369_silu_matmul_abs(const Tensor&,const Tensor&);
Tensor fused_1369_silu_matmul_abs_contiguous(const Tensor&,const Tensor&);
Tensor fused_1369_silu_matmul_abs_cpu(const Tensor&,const Tensor&);
Tensor fused_1369_silu_matmul_abs_clone(const Tensor&,const Tensor&);
Tensor fused_1369_silu_matmul_abs_detached(const Tensor&,const Tensor&);
bool fused_1369_silu_matmul_abs_same_shape(const Tensor&,const Tensor&);
int64_t fused_1369_silu_matmul_abs_numel(const Tensor&);
std::string fused_1369_silu_matmul_abs_shape(const Tensor&);
const char* fused_1369_silu_matmul_abs_name();
}

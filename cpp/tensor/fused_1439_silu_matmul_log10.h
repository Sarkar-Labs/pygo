#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1439_silu_matmul_log10(const Tensor&,const Tensor&);
Tensor fused_1439_silu_matmul_log10_contiguous(const Tensor&,const Tensor&);
Tensor fused_1439_silu_matmul_log10_cpu(const Tensor&,const Tensor&);
Tensor fused_1439_silu_matmul_log10_clone(const Tensor&,const Tensor&);
Tensor fused_1439_silu_matmul_log10_detached(const Tensor&,const Tensor&);
bool fused_1439_silu_matmul_log10_same_shape(const Tensor&,const Tensor&);
int64_t fused_1439_silu_matmul_log10_numel(const Tensor&);
std::string fused_1439_silu_matmul_log10_shape(const Tensor&);
const char* fused_1439_silu_matmul_log10_name();
}

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1634_log10_matmul_tanh(const Tensor&,const Tensor&);
Tensor fused_1634_log10_matmul_tanh_contiguous(const Tensor&,const Tensor&);
Tensor fused_1634_log10_matmul_tanh_cpu(const Tensor&,const Tensor&);
Tensor fused_1634_log10_matmul_tanh_clone(const Tensor&,const Tensor&);
Tensor fused_1634_log10_matmul_tanh_detached(const Tensor&,const Tensor&);
bool fused_1634_log10_matmul_tanh_same_shape(const Tensor&,const Tensor&);
int64_t fused_1634_log10_matmul_tanh_numel(const Tensor&);
std::string fused_1634_log10_matmul_tanh_shape(const Tensor&);
const char* fused_1634_log10_matmul_tanh_name();
}

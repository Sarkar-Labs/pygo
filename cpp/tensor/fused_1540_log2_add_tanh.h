#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1540_log2_add_tanh(const Tensor&,const Tensor&);
Tensor fused_1540_log2_add_tanh_contiguous(const Tensor&,const Tensor&);
Tensor fused_1540_log2_add_tanh_cpu(const Tensor&,const Tensor&);
Tensor fused_1540_log2_add_tanh_clone(const Tensor&,const Tensor&);
Tensor fused_1540_log2_add_tanh_detached(const Tensor&,const Tensor&);
bool fused_1540_log2_add_tanh_same_shape(const Tensor&,const Tensor&);
int64_t fused_1540_log2_add_tanh_numel(const Tensor&);
std::string fused_1540_log2_add_tanh_shape(const Tensor&);
const char* fused_1540_log2_add_tanh_name();
}

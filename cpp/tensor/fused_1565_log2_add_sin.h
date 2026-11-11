#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1565_log2_add_sin(const Tensor&,const Tensor&);
Tensor fused_1565_log2_add_sin_contiguous(const Tensor&,const Tensor&);
Tensor fused_1565_log2_add_sin_cpu(const Tensor&,const Tensor&);
Tensor fused_1565_log2_add_sin_clone(const Tensor&,const Tensor&);
Tensor fused_1565_log2_add_sin_detached(const Tensor&,const Tensor&);
bool fused_1565_log2_add_sin_same_shape(const Tensor&,const Tensor&);
int64_t fused_1565_log2_add_sin_numel(const Tensor&);
std::string fused_1565_log2_add_sin_shape(const Tensor&);
const char* fused_1565_log2_add_sin_name();
}

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1042_ceil_mul_ceil(const Tensor&,const Tensor&);
Tensor fused_1042_ceil_mul_ceil_contiguous(const Tensor&,const Tensor&);
Tensor fused_1042_ceil_mul_ceil_cpu(const Tensor&,const Tensor&);
Tensor fused_1042_ceil_mul_ceil_clone(const Tensor&,const Tensor&);
Tensor fused_1042_ceil_mul_ceil_detached(const Tensor&,const Tensor&);
bool fused_1042_ceil_mul_ceil_same_shape(const Tensor&,const Tensor&);
int64_t fused_1042_ceil_mul_ceil_numel(const Tensor&);
std::string fused_1042_ceil_mul_ceil_shape(const Tensor&);
const char* fused_1042_ceil_mul_ceil_name();
}

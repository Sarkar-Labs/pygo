#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0915_floor_add_abs(const Tensor&,const Tensor&);
Tensor fused_0915_floor_add_abs_contiguous(const Tensor&,const Tensor&);
Tensor fused_0915_floor_add_abs_cpu(const Tensor&,const Tensor&);
Tensor fused_0915_floor_add_abs_clone(const Tensor&,const Tensor&);
Tensor fused_0915_floor_add_abs_detached(const Tensor&,const Tensor&);
bool fused_0915_floor_add_abs_same_shape(const Tensor&,const Tensor&);
int64_t fused_0915_floor_add_abs_numel(const Tensor&);
std::string fused_0915_floor_add_abs_shape(const Tensor&);
const char* fused_0915_floor_add_abs_name();
}

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0971_floor_sub_silu(const Tensor&,const Tensor&);
Tensor fused_0971_floor_sub_silu_contiguous(const Tensor&,const Tensor&);
Tensor fused_0971_floor_sub_silu_cpu(const Tensor&,const Tensor&);
Tensor fused_0971_floor_sub_silu_clone(const Tensor&,const Tensor&);
Tensor fused_0971_floor_sub_silu_detached(const Tensor&,const Tensor&);
bool fused_0971_floor_sub_silu_same_shape(const Tensor&,const Tensor&);
int64_t fused_0971_floor_sub_silu_numel(const Tensor&);
std::string fused_0971_floor_sub_silu_shape(const Tensor&);
const char* fused_0971_floor_sub_silu_name();
}

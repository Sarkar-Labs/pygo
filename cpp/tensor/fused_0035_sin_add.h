#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0035_sin_add(const Tensor&,const Tensor&);
Tensor fused_0035_sin_add_contiguous(const Tensor&,const Tensor&);
Tensor fused_0035_sin_add_cpu(const Tensor&,const Tensor&);
Tensor fused_0035_sin_add_clone(const Tensor&,const Tensor&);
Tensor fused_0035_sin_add_detached(const Tensor&,const Tensor&);
bool fused_0035_sin_add_same_shape(const Tensor&,const Tensor&);
int64_t fused_0035_sin_add_numel(const Tensor&);
std::string fused_0035_sin_add_shape(const Tensor&);
const char* fused_0035_sin_add_name();
}

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1220_sign_add_ceil(const Tensor&,const Tensor&);
Tensor fused_1220_sign_add_ceil_contiguous(const Tensor&,const Tensor&);
Tensor fused_1220_sign_add_ceil_cpu(const Tensor&,const Tensor&);
Tensor fused_1220_sign_add_ceil_clone(const Tensor&,const Tensor&);
Tensor fused_1220_sign_add_ceil_detached(const Tensor&,const Tensor&);
bool fused_1220_sign_add_ceil_same_shape(const Tensor&,const Tensor&);
int64_t fused_1220_sign_add_ceil_numel(const Tensor&);
std::string fused_1220_sign_add_ceil_shape(const Tensor&);
const char* fused_1220_sign_add_ceil_name();
}

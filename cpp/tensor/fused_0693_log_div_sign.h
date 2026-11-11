#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0693_log_div_sign(const Tensor&,const Tensor&);
Tensor fused_0693_log_div_sign_contiguous(const Tensor&,const Tensor&);
Tensor fused_0693_log_div_sign_cpu(const Tensor&,const Tensor&);
Tensor fused_0693_log_div_sign_clone(const Tensor&,const Tensor&);
Tensor fused_0693_log_div_sign_detached(const Tensor&,const Tensor&);
bool fused_0693_log_div_sign_same_shape(const Tensor&,const Tensor&);
int64_t fused_0693_log_div_sign_numel(const Tensor&);
std::string fused_0693_log_div_sign_shape(const Tensor&);
const char* fused_0693_log_div_sign_name();
}

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0122_relu_mul_log(const Tensor&,const Tensor&);
Tensor fused_0122_relu_mul_log_contiguous(const Tensor&,const Tensor&);
Tensor fused_0122_relu_mul_log_cpu(const Tensor&,const Tensor&);
Tensor fused_0122_relu_mul_log_clone(const Tensor&,const Tensor&);
Tensor fused_0122_relu_mul_log_detached(const Tensor&,const Tensor&);
bool fused_0122_relu_mul_log_same_shape(const Tensor&,const Tensor&);
int64_t fused_0122_relu_mul_log_numel(const Tensor&);
std::string fused_0122_relu_mul_log_shape(const Tensor&);
const char* fused_0122_relu_mul_log_name();
}

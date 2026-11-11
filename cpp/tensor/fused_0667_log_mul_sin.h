#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0667_log_mul_sin(const Tensor&,const Tensor&);
Tensor fused_0667_log_mul_sin_contiguous(const Tensor&,const Tensor&);
Tensor fused_0667_log_mul_sin_cpu(const Tensor&,const Tensor&);
Tensor fused_0667_log_mul_sin_clone(const Tensor&,const Tensor&);
Tensor fused_0667_log_mul_sin_detached(const Tensor&,const Tensor&);
bool fused_0667_log_mul_sin_same_shape(const Tensor&,const Tensor&);
int64_t fused_0667_log_mul_sin_numel(const Tensor&);
std::string fused_0667_log_mul_sin_shape(const Tensor&);
const char* fused_0667_log_mul_sin_name();
}

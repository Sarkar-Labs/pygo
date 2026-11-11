#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1662_log10_mul_cos(const Tensor&,const Tensor&);
Tensor fused_1662_log10_mul_cos_contiguous(const Tensor&,const Tensor&);
Tensor fused_1662_log10_mul_cos_cpu(const Tensor&,const Tensor&);
Tensor fused_1662_log10_mul_cos_clone(const Tensor&,const Tensor&);
Tensor fused_1662_log10_mul_cos_detached(const Tensor&,const Tensor&);
bool fused_1662_log10_mul_cos_same_shape(const Tensor&,const Tensor&);
int64_t fused_1662_log10_mul_cos_numel(const Tensor&);
std::string fused_1662_log10_mul_cos_shape(const Tensor&);
const char* fused_1662_log10_mul_cos_name();
}

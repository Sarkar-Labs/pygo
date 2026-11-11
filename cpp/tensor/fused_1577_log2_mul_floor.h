#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1577_log2_mul_floor(const Tensor&,const Tensor&);
Tensor fused_1577_log2_mul_floor_contiguous(const Tensor&,const Tensor&);
Tensor fused_1577_log2_mul_floor_cpu(const Tensor&,const Tensor&);
Tensor fused_1577_log2_mul_floor_clone(const Tensor&,const Tensor&);
Tensor fused_1577_log2_mul_floor_detached(const Tensor&,const Tensor&);
bool fused_1577_log2_mul_floor_same_shape(const Tensor&,const Tensor&);
int64_t fused_1577_log2_mul_floor_numel(const Tensor&);
std::string fused_1577_log2_mul_floor_shape(const Tensor&);
const char* fused_1577_log2_mul_floor_name();
}

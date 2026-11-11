#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0769_sin_matmul_floor(const Tensor&,const Tensor&);
Tensor fused_0769_sin_matmul_floor_contiguous(const Tensor&,const Tensor&);
Tensor fused_0769_sin_matmul_floor_cpu(const Tensor&,const Tensor&);
Tensor fused_0769_sin_matmul_floor_clone(const Tensor&,const Tensor&);
Tensor fused_0769_sin_matmul_floor_detached(const Tensor&,const Tensor&);
bool fused_0769_sin_matmul_floor_same_shape(const Tensor&,const Tensor&);
int64_t fused_0769_sin_matmul_floor_numel(const Tensor&);
std::string fused_0769_sin_matmul_floor_shape(const Tensor&);
const char* fused_0769_sin_matmul_floor_name();
}

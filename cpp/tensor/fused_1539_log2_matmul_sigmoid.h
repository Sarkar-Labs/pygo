#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1539_log2_matmul_sigmoid(const Tensor&,const Tensor&);
Tensor fused_1539_log2_matmul_sigmoid_contiguous(const Tensor&,const Tensor&);
Tensor fused_1539_log2_matmul_sigmoid_cpu(const Tensor&,const Tensor&);
Tensor fused_1539_log2_matmul_sigmoid_clone(const Tensor&,const Tensor&);
Tensor fused_1539_log2_matmul_sigmoid_detached(const Tensor&,const Tensor&);
bool fused_1539_log2_matmul_sigmoid_same_shape(const Tensor&,const Tensor&);
int64_t fused_1539_log2_matmul_sigmoid_numel(const Tensor&);
std::string fused_1539_log2_matmul_sigmoid_shape(const Tensor&);
const char* fused_1539_log2_matmul_sigmoid_name();
}

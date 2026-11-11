#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_0264_sigmoid_matmul_log2(const Tensor&,const Tensor&);
Tensor fused_0264_sigmoid_matmul_log2_contiguous(const Tensor&,const Tensor&);
Tensor fused_0264_sigmoid_matmul_log2_cpu(const Tensor&,const Tensor&);
Tensor fused_0264_sigmoid_matmul_log2_clone(const Tensor&,const Tensor&);
Tensor fused_0264_sigmoid_matmul_log2_detached(const Tensor&,const Tensor&);
bool fused_0264_sigmoid_matmul_log2_same_shape(const Tensor&,const Tensor&);
int64_t fused_0264_sigmoid_matmul_log2_numel(const Tensor&);
std::string fused_0264_sigmoid_matmul_log2_shape(const Tensor&);
const char* fused_0264_sigmoid_matmul_log2_name();
}

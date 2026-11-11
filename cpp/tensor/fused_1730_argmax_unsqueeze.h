#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1730_argmax_unsqueeze(const Tensor&,const Tensor&);
Tensor fused_1730_argmax_unsqueeze_contiguous(const Tensor&,const Tensor&);
Tensor fused_1730_argmax_unsqueeze_cpu(const Tensor&,const Tensor&);
Tensor fused_1730_argmax_unsqueeze_clone(const Tensor&,const Tensor&);
Tensor fused_1730_argmax_unsqueeze_detached(const Tensor&,const Tensor&);
bool fused_1730_argmax_unsqueeze_same_shape(const Tensor&,const Tensor&);
int64_t fused_1730_argmax_unsqueeze_numel(const Tensor&);
std::string fused_1730_argmax_unsqueeze_shape(const Tensor&);
const char* fused_1730_argmax_unsqueeze_name();
}

#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1732_argmax_detach(const Tensor&,const Tensor&);
Tensor fused_1732_argmax_detach_contiguous(const Tensor&,const Tensor&);
Tensor fused_1732_argmax_detach_cpu(const Tensor&,const Tensor&);
Tensor fused_1732_argmax_detach_clone(const Tensor&,const Tensor&);
Tensor fused_1732_argmax_detach_detached(const Tensor&,const Tensor&);
bool fused_1732_argmax_detach_same_shape(const Tensor&,const Tensor&);
int64_t fused_1732_argmax_detach_numel(const Tensor&);
std::string fused_1732_argmax_detach_shape(const Tensor&);
const char* fused_1732_argmax_detach_name();
}

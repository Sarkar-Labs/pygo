#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1729_argmax_squeeze(const Tensor&,const Tensor&);
Tensor fused_1729_argmax_squeeze_contiguous(const Tensor&,const Tensor&);
Tensor fused_1729_argmax_squeeze_cpu(const Tensor&,const Tensor&);
Tensor fused_1729_argmax_squeeze_clone(const Tensor&,const Tensor&);
Tensor fused_1729_argmax_squeeze_detached(const Tensor&,const Tensor&);
bool fused_1729_argmax_squeeze_same_shape(const Tensor&,const Tensor&);
int64_t fused_1729_argmax_squeeze_numel(const Tensor&);
std::string fused_1729_argmax_squeeze_shape(const Tensor&);
const char* fused_1729_argmax_squeeze_name();
}

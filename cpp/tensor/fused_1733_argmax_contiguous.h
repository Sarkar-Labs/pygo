#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1733_argmax_contiguous(const Tensor&,const Tensor&);
Tensor fused_1733_argmax_contiguous_contiguous(const Tensor&,const Tensor&);
Tensor fused_1733_argmax_contiguous_cpu(const Tensor&,const Tensor&);
Tensor fused_1733_argmax_contiguous_clone(const Tensor&,const Tensor&);
Tensor fused_1733_argmax_contiguous_detached(const Tensor&,const Tensor&);
bool fused_1733_argmax_contiguous_same_shape(const Tensor&,const Tensor&);
int64_t fused_1733_argmax_contiguous_numel(const Tensor&);
std::string fused_1733_argmax_contiguous_shape(const Tensor&);
const char* fused_1733_argmax_contiguous_name();
}

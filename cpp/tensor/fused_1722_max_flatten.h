#pragma once
#include "tensor.h"
#include <string>
namespace pygo::tensor_ops {
Tensor fused_1722_max_flatten(const Tensor&,const Tensor&);
Tensor fused_1722_max_flatten_contiguous(const Tensor&,const Tensor&);
Tensor fused_1722_max_flatten_cpu(const Tensor&,const Tensor&);
Tensor fused_1722_max_flatten_clone(const Tensor&,const Tensor&);
Tensor fused_1722_max_flatten_detached(const Tensor&,const Tensor&);
bool fused_1722_max_flatten_same_shape(const Tensor&,const Tensor&);
int64_t fused_1722_max_flatten_numel(const Tensor&);
std::string fused_1722_max_flatten_shape(const Tensor&);
const char* fused_1722_max_flatten_name();
}

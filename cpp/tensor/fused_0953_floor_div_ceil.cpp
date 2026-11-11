#include "fused_0953_floor_div_ceil.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0953_floor_div_ceil(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("floor_div_ceil: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("floor_div_ceil: device mismatch");
}
Tensor fused_0953_floor_div_ceil(const Tensor& x,const Tensor& y) { validate_fused_0953_floor_div_ceil(x,y); Tensor r=((x / y).floor()).ceil(); if(!r.defined()) throw std::runtime_error("floor_div_ceil: undefined result"); return r; }
Tensor fused_0953_floor_div_ceil_contiguous(const Tensor& x,const Tensor& y) { return fused_0953_floor_div_ceil(x.contiguous(),y.contiguous()); }
Tensor fused_0953_floor_div_ceil_cpu(const Tensor& x,const Tensor& y) { return fused_0953_floor_div_ceil(x.cpu(),y.cpu()); }
Tensor fused_0953_floor_div_ceil_clone(const Tensor& x,const Tensor& y) { return fused_0953_floor_div_ceil(x.clone(),y.clone()); }
Tensor fused_0953_floor_div_ceil_detached(const Tensor& x,const Tensor& y) { return fused_0953_floor_div_ceil(x.detach(),y.detach()); }
bool fused_0953_floor_div_ceil_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0953_floor_div_ceil_numel(const Tensor& x) { return x.numel(); }
std::string fused_0953_floor_div_ceil_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0953_floor_div_ceil_name() { return "floor_div_ceil"; }
}

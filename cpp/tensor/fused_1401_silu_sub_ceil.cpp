#include "fused_1401_silu_sub_ceil.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1401_silu_sub_ceil(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("silu_sub_ceil: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("silu_sub_ceil: device mismatch");
}
Tensor fused_1401_silu_sub_ceil(const Tensor& x,const Tensor& y) { validate_fused_1401_silu_sub_ceil(x,y); Tensor r=((x - y).silu()).ceil(); if(!r.defined()) throw std::runtime_error("silu_sub_ceil: undefined result"); return r; }
Tensor fused_1401_silu_sub_ceil_contiguous(const Tensor& x,const Tensor& y) { return fused_1401_silu_sub_ceil(x.contiguous(),y.contiguous()); }
Tensor fused_1401_silu_sub_ceil_cpu(const Tensor& x,const Tensor& y) { return fused_1401_silu_sub_ceil(x.cpu(),y.cpu()); }
Tensor fused_1401_silu_sub_ceil_clone(const Tensor& x,const Tensor& y) { return fused_1401_silu_sub_ceil(x.clone(),y.clone()); }
Tensor fused_1401_silu_sub_ceil_detached(const Tensor& x,const Tensor& y) { return fused_1401_silu_sub_ceil(x.detach(),y.detach()); }
bool fused_1401_silu_sub_ceil_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1401_silu_sub_ceil_numel(const Tensor& x) { return x.numel(); }
std::string fused_1401_silu_sub_ceil_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1401_silu_sub_ceil_name() { return "silu_sub_ceil"; }
}

#include "fused_0466_sqrt_sub_abs.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0466_sqrt_sub_abs(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sqrt_sub_abs: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sqrt_sub_abs: device mismatch");
}
Tensor fused_0466_sqrt_sub_abs(const Tensor& x,const Tensor& y) { validate_fused_0466_sqrt_sub_abs(x,y); Tensor r=((x - y).sqrt()).abs(); if(!r.defined()) throw std::runtime_error("sqrt_sub_abs: undefined result"); return r; }
Tensor fused_0466_sqrt_sub_abs_contiguous(const Tensor& x,const Tensor& y) { return fused_0466_sqrt_sub_abs(x.contiguous(),y.contiguous()); }
Tensor fused_0466_sqrt_sub_abs_cpu(const Tensor& x,const Tensor& y) { return fused_0466_sqrt_sub_abs(x.cpu(),y.cpu()); }
Tensor fused_0466_sqrt_sub_abs_clone(const Tensor& x,const Tensor& y) { return fused_0466_sqrt_sub_abs(x.clone(),y.clone()); }
Tensor fused_0466_sqrt_sub_abs_detached(const Tensor& x,const Tensor& y) { return fused_0466_sqrt_sub_abs(x.detach(),y.detach()); }
bool fused_0466_sqrt_sub_abs_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0466_sqrt_sub_abs_numel(const Tensor& x) { return x.numel(); }
std::string fused_0466_sqrt_sub_abs_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0466_sqrt_sub_abs_name() { return "sqrt_sub_abs"; }
}

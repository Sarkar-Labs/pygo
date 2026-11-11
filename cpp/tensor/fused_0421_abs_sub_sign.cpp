#include "fused_0421_abs_sub_sign.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0421_abs_sub_sign(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("abs_sub_sign: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("abs_sub_sign: device mismatch");
}
Tensor fused_0421_abs_sub_sign(const Tensor& x,const Tensor& y) { validate_fused_0421_abs_sub_sign(x,y); Tensor r=((x - y).abs()).sign(); if(!r.defined()) throw std::runtime_error("abs_sub_sign: undefined result"); return r; }
Tensor fused_0421_abs_sub_sign_contiguous(const Tensor& x,const Tensor& y) { return fused_0421_abs_sub_sign(x.contiguous(),y.contiguous()); }
Tensor fused_0421_abs_sub_sign_cpu(const Tensor& x,const Tensor& y) { return fused_0421_abs_sub_sign(x.cpu(),y.cpu()); }
Tensor fused_0421_abs_sub_sign_clone(const Tensor& x,const Tensor& y) { return fused_0421_abs_sub_sign(x.clone(),y.clone()); }
Tensor fused_0421_abs_sub_sign_detached(const Tensor& x,const Tensor& y) { return fused_0421_abs_sub_sign(x.detach(),y.detach()); }
bool fused_0421_abs_sub_sign_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0421_abs_sub_sign_numel(const Tensor& x) { return x.numel(); }
std::string fused_0421_abs_sub_sign_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0421_abs_sub_sign_name() { return "abs_sub_sign"; }
}

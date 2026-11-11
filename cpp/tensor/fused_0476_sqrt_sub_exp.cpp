#include "fused_0476_sqrt_sub_exp.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0476_sqrt_sub_exp(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sqrt_sub_exp: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sqrt_sub_exp: device mismatch");
}
Tensor fused_0476_sqrt_sub_exp(const Tensor& x,const Tensor& y) { validate_fused_0476_sqrt_sub_exp(x,y); Tensor r=((x - y).sqrt()).exp(); if(!r.defined()) throw std::runtime_error("sqrt_sub_exp: undefined result"); return r; }
Tensor fused_0476_sqrt_sub_exp_contiguous(const Tensor& x,const Tensor& y) { return fused_0476_sqrt_sub_exp(x.contiguous(),y.contiguous()); }
Tensor fused_0476_sqrt_sub_exp_cpu(const Tensor& x,const Tensor& y) { return fused_0476_sqrt_sub_exp(x.cpu(),y.cpu()); }
Tensor fused_0476_sqrt_sub_exp_clone(const Tensor& x,const Tensor& y) { return fused_0476_sqrt_sub_exp(x.clone(),y.clone()); }
Tensor fused_0476_sqrt_sub_exp_detached(const Tensor& x,const Tensor& y) { return fused_0476_sqrt_sub_exp(x.detach(),y.detach()); }
bool fused_0476_sqrt_sub_exp_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0476_sqrt_sub_exp_numel(const Tensor& x) { return x.numel(); }
std::string fused_0476_sqrt_sub_exp_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0476_sqrt_sub_exp_name() { return "sqrt_sub_exp"; }
}

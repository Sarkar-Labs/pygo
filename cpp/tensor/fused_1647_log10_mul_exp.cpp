#include "fused_1647_log10_mul_exp.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1647_log10_mul_exp(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log10_mul_exp: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log10_mul_exp: device mismatch");
}
Tensor fused_1647_log10_mul_exp(const Tensor& x,const Tensor& y) { validate_fused_1647_log10_mul_exp(x,y); Tensor r=((x * y).log10()).exp(); if(!r.defined()) throw std::runtime_error("log10_mul_exp: undefined result"); return r; }
Tensor fused_1647_log10_mul_exp_contiguous(const Tensor& x,const Tensor& y) { return fused_1647_log10_mul_exp(x.contiguous(),y.contiguous()); }
Tensor fused_1647_log10_mul_exp_cpu(const Tensor& x,const Tensor& y) { return fused_1647_log10_mul_exp(x.cpu(),y.cpu()); }
Tensor fused_1647_log10_mul_exp_clone(const Tensor& x,const Tensor& y) { return fused_1647_log10_mul_exp(x.clone(),y.clone()); }
Tensor fused_1647_log10_mul_exp_detached(const Tensor& x,const Tensor& y) { return fused_1647_log10_mul_exp(x.detach(),y.detach()); }
bool fused_1647_log10_mul_exp_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1647_log10_mul_exp_numel(const Tensor& x) { return x.numel(); }
std::string fused_1647_log10_mul_exp_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1647_log10_mul_exp_name() { return "log10_mul_exp"; }
}

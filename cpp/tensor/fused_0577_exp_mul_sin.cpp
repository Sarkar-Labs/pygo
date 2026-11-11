#include "fused_0577_exp_mul_sin.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0577_exp_mul_sin(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("exp_mul_sin: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("exp_mul_sin: device mismatch");
}
Tensor fused_0577_exp_mul_sin(const Tensor& x,const Tensor& y) { validate_fused_0577_exp_mul_sin(x,y); Tensor r=((x * y).exp()).sin(); if(!r.defined()) throw std::runtime_error("exp_mul_sin: undefined result"); return r; }
Tensor fused_0577_exp_mul_sin_contiguous(const Tensor& x,const Tensor& y) { return fused_0577_exp_mul_sin(x.contiguous(),y.contiguous()); }
Tensor fused_0577_exp_mul_sin_cpu(const Tensor& x,const Tensor& y) { return fused_0577_exp_mul_sin(x.cpu(),y.cpu()); }
Tensor fused_0577_exp_mul_sin_clone(const Tensor& x,const Tensor& y) { return fused_0577_exp_mul_sin(x.clone(),y.clone()); }
Tensor fused_0577_exp_mul_sin_detached(const Tensor& x,const Tensor& y) { return fused_0577_exp_mul_sin(x.detach(),y.detach()); }
bool fused_0577_exp_mul_sin_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0577_exp_mul_sin_numel(const Tensor& x) { return x.numel(); }
std::string fused_0577_exp_mul_sin_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0577_exp_mul_sin_name() { return "exp_mul_sin"; }
}

#include "fused_0028_exp_div.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0028_exp_div(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("exp_div: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("exp_div: device mismatch");
}
Tensor fused_0028_exp_div(const Tensor& x,const Tensor& y) { validate_fused_0028_exp_div(x,y); Tensor r=(x / y).exp(); if(!r.defined()) throw std::runtime_error("exp_div: undefined result"); return r; }
Tensor fused_0028_exp_div_contiguous(const Tensor& x,const Tensor& y) { return fused_0028_exp_div(x.contiguous(),y.contiguous()); }
Tensor fused_0028_exp_div_cpu(const Tensor& x,const Tensor& y) { return fused_0028_exp_div(x.cpu(),y.cpu()); }
Tensor fused_0028_exp_div_clone(const Tensor& x,const Tensor& y) { return fused_0028_exp_div(x.clone(),y.clone()); }
Tensor fused_0028_exp_div_detached(const Tensor& x,const Tensor& y) { return fused_0028_exp_div(x.detach(),y.detach()); }
bool fused_0028_exp_div_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0028_exp_div_numel(const Tensor& x) { return x.numel(); }
std::string fused_0028_exp_div_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0028_exp_div_name() { return "exp_div"; }
}

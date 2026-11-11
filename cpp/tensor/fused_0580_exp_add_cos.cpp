#include "fused_0580_exp_add_cos.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0580_exp_add_cos(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("exp_add_cos: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("exp_add_cos: device mismatch");
}
Tensor fused_0580_exp_add_cos(const Tensor& x,const Tensor& y) { validate_fused_0580_exp_add_cos(x,y); Tensor r=((x + y).exp()).cos(); if(!r.defined()) throw std::runtime_error("exp_add_cos: undefined result"); return r; }
Tensor fused_0580_exp_add_cos_contiguous(const Tensor& x,const Tensor& y) { return fused_0580_exp_add_cos(x.contiguous(),y.contiguous()); }
Tensor fused_0580_exp_add_cos_cpu(const Tensor& x,const Tensor& y) { return fused_0580_exp_add_cos(x.cpu(),y.cpu()); }
Tensor fused_0580_exp_add_cos_clone(const Tensor& x,const Tensor& y) { return fused_0580_exp_add_cos(x.clone(),y.clone()); }
Tensor fused_0580_exp_add_cos_detached(const Tensor& x,const Tensor& y) { return fused_0580_exp_add_cos(x.detach(),y.detach()); }
bool fused_0580_exp_add_cos_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0580_exp_add_cos_numel(const Tensor& x) { return x.numel(); }
std::string fused_0580_exp_add_cos_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0580_exp_add_cos_name() { return "exp_add_cos"; }
}

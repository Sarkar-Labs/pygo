#include "fused_1500_neg_add_sign.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1500_neg_add_sign(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("neg_add_sign: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("neg_add_sign: device mismatch");
}
Tensor fused_1500_neg_add_sign(const Tensor& x,const Tensor& y) { validate_fused_1500_neg_add_sign(x,y); Tensor r=((x + y).neg()).sign(); if(!r.defined()) throw std::runtime_error("neg_add_sign: undefined result"); return r; }
Tensor fused_1500_neg_add_sign_contiguous(const Tensor& x,const Tensor& y) { return fused_1500_neg_add_sign(x.contiguous(),y.contiguous()); }
Tensor fused_1500_neg_add_sign_cpu(const Tensor& x,const Tensor& y) { return fused_1500_neg_add_sign(x.cpu(),y.cpu()); }
Tensor fused_1500_neg_add_sign_clone(const Tensor& x,const Tensor& y) { return fused_1500_neg_add_sign(x.clone(),y.clone()); }
Tensor fused_1500_neg_add_sign_detached(const Tensor& x,const Tensor& y) { return fused_1500_neg_add_sign(x.detach(),y.detach()); }
bool fused_1500_neg_add_sign_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1500_neg_add_sign_numel(const Tensor& x) { return x.numel(); }
std::string fused_1500_neg_add_sign_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1500_neg_add_sign_name() { return "neg_add_sign"; }
}

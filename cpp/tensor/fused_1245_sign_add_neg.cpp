#include "fused_1245_sign_add_neg.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1245_sign_add_neg(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sign_add_neg: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sign_add_neg: device mismatch");
}
Tensor fused_1245_sign_add_neg(const Tensor& x,const Tensor& y) { validate_fused_1245_sign_add_neg(x,y); Tensor r=((x + y).sign()).neg(); if(!r.defined()) throw std::runtime_error("sign_add_neg: undefined result"); return r; }
Tensor fused_1245_sign_add_neg_contiguous(const Tensor& x,const Tensor& y) { return fused_1245_sign_add_neg(x.contiguous(),y.contiguous()); }
Tensor fused_1245_sign_add_neg_cpu(const Tensor& x,const Tensor& y) { return fused_1245_sign_add_neg(x.cpu(),y.cpu()); }
Tensor fused_1245_sign_add_neg_clone(const Tensor& x,const Tensor& y) { return fused_1245_sign_add_neg(x.clone(),y.clone()); }
Tensor fused_1245_sign_add_neg_detached(const Tensor& x,const Tensor& y) { return fused_1245_sign_add_neg(x.detach(),y.detach()); }
bool fused_1245_sign_add_neg_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1245_sign_add_neg_numel(const Tensor& x) { return x.numel(); }
std::string fused_1245_sign_add_neg_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1245_sign_add_neg_name() { return "sign_add_neg"; }
}

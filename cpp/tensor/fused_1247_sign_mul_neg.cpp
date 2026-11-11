#include "fused_1247_sign_mul_neg.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1247_sign_mul_neg(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sign_mul_neg: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sign_mul_neg: device mismatch");
}
Tensor fused_1247_sign_mul_neg(const Tensor& x,const Tensor& y) { validate_fused_1247_sign_mul_neg(x,y); Tensor r=((x * y).sign()).neg(); if(!r.defined()) throw std::runtime_error("sign_mul_neg: undefined result"); return r; }
Tensor fused_1247_sign_mul_neg_contiguous(const Tensor& x,const Tensor& y) { return fused_1247_sign_mul_neg(x.contiguous(),y.contiguous()); }
Tensor fused_1247_sign_mul_neg_cpu(const Tensor& x,const Tensor& y) { return fused_1247_sign_mul_neg(x.cpu(),y.cpu()); }
Tensor fused_1247_sign_mul_neg_clone(const Tensor& x,const Tensor& y) { return fused_1247_sign_mul_neg(x.clone(),y.clone()); }
Tensor fused_1247_sign_mul_neg_detached(const Tensor& x,const Tensor& y) { return fused_1247_sign_mul_neg(x.detach(),y.detach()); }
bool fused_1247_sign_mul_neg_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1247_sign_mul_neg_numel(const Tensor& x) { return x.numel(); }
std::string fused_1247_sign_mul_neg_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1247_sign_mul_neg_name() { return "sign_mul_neg"; }
}

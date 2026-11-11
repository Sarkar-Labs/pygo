#include "fused_1502_neg_mul_sign.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1502_neg_mul_sign(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("neg_mul_sign: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("neg_mul_sign: device mismatch");
}
Tensor fused_1502_neg_mul_sign(const Tensor& x,const Tensor& y) { validate_fused_1502_neg_mul_sign(x,y); Tensor r=((x * y).neg()).sign(); if(!r.defined()) throw std::runtime_error("neg_mul_sign: undefined result"); return r; }
Tensor fused_1502_neg_mul_sign_contiguous(const Tensor& x,const Tensor& y) { return fused_1502_neg_mul_sign(x.contiguous(),y.contiguous()); }
Tensor fused_1502_neg_mul_sign_cpu(const Tensor& x,const Tensor& y) { return fused_1502_neg_mul_sign(x.cpu(),y.cpu()); }
Tensor fused_1502_neg_mul_sign_clone(const Tensor& x,const Tensor& y) { return fused_1502_neg_mul_sign(x.clone(),y.clone()); }
Tensor fused_1502_neg_mul_sign_detached(const Tensor& x,const Tensor& y) { return fused_1502_neg_mul_sign(x.detach(),y.detach()); }
bool fused_1502_neg_mul_sign_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1502_neg_mul_sign_numel(const Tensor& x) { return x.numel(); }
std::string fused_1502_neg_mul_sign_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1502_neg_mul_sign_name() { return "neg_mul_sign"; }
}

#include "fused_1607_log2_mul_neg.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1607_log2_mul_neg(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log2_mul_neg: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log2_mul_neg: device mismatch");
}
Tensor fused_1607_log2_mul_neg(const Tensor& x,const Tensor& y) { validate_fused_1607_log2_mul_neg(x,y); Tensor r=((x * y).log2()).neg(); if(!r.defined()) throw std::runtime_error("log2_mul_neg: undefined result"); return r; }
Tensor fused_1607_log2_mul_neg_contiguous(const Tensor& x,const Tensor& y) { return fused_1607_log2_mul_neg(x.contiguous(),y.contiguous()); }
Tensor fused_1607_log2_mul_neg_cpu(const Tensor& x,const Tensor& y) { return fused_1607_log2_mul_neg(x.cpu(),y.cpu()); }
Tensor fused_1607_log2_mul_neg_clone(const Tensor& x,const Tensor& y) { return fused_1607_log2_mul_neg(x.clone(),y.clone()); }
Tensor fused_1607_log2_mul_neg_detached(const Tensor& x,const Tensor& y) { return fused_1607_log2_mul_neg(x.detach(),y.detach()); }
bool fused_1607_log2_mul_neg_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1607_log2_mul_neg_numel(const Tensor& x) { return x.numel(); }
std::string fused_1607_log2_mul_neg_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1607_log2_mul_neg_name() { return "log2_mul_neg"; }
}

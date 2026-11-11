#include "fused_1336_gelu_sub_neg.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1336_gelu_sub_neg(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("gelu_sub_neg: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("gelu_sub_neg: device mismatch");
}
Tensor fused_1336_gelu_sub_neg(const Tensor& x,const Tensor& y) { validate_fused_1336_gelu_sub_neg(x,y); Tensor r=((x - y).gelu()).neg(); if(!r.defined()) throw std::runtime_error("gelu_sub_neg: undefined result"); return r; }
Tensor fused_1336_gelu_sub_neg_contiguous(const Tensor& x,const Tensor& y) { return fused_1336_gelu_sub_neg(x.contiguous(),y.contiguous()); }
Tensor fused_1336_gelu_sub_neg_cpu(const Tensor& x,const Tensor& y) { return fused_1336_gelu_sub_neg(x.cpu(),y.cpu()); }
Tensor fused_1336_gelu_sub_neg_clone(const Tensor& x,const Tensor& y) { return fused_1336_gelu_sub_neg(x.clone(),y.clone()); }
Tensor fused_1336_gelu_sub_neg_detached(const Tensor& x,const Tensor& y) { return fused_1336_gelu_sub_neg(x.detach(),y.detach()); }
bool fused_1336_gelu_sub_neg_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1336_gelu_sub_neg_numel(const Tensor& x) { return x.numel(); }
std::string fused_1336_gelu_sub_neg_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1336_gelu_sub_neg_name() { return "gelu_sub_neg"; }
}

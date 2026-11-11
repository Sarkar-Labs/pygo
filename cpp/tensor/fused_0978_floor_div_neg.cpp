#include "fused_0978_floor_div_neg.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0978_floor_div_neg(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("floor_div_neg: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("floor_div_neg: device mismatch");
}
Tensor fused_0978_floor_div_neg(const Tensor& x,const Tensor& y) { validate_fused_0978_floor_div_neg(x,y); Tensor r=((x / y).floor()).neg(); if(!r.defined()) throw std::runtime_error("floor_div_neg: undefined result"); return r; }
Tensor fused_0978_floor_div_neg_contiguous(const Tensor& x,const Tensor& y) { return fused_0978_floor_div_neg(x.contiguous(),y.contiguous()); }
Tensor fused_0978_floor_div_neg_cpu(const Tensor& x,const Tensor& y) { return fused_0978_floor_div_neg(x.cpu(),y.cpu()); }
Tensor fused_0978_floor_div_neg_clone(const Tensor& x,const Tensor& y) { return fused_0978_floor_div_neg(x.clone(),y.clone()); }
Tensor fused_0978_floor_div_neg_detached(const Tensor& x,const Tensor& y) { return fused_0978_floor_div_neg(x.detach(),y.detach()); }
bool fused_0978_floor_div_neg_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0978_floor_div_neg_numel(const Tensor& x) { return x.numel(); }
std::string fused_0978_floor_div_neg_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0978_floor_div_neg_name() { return "floor_div_neg"; }
}

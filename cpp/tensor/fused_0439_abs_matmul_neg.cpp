#include "fused_0439_abs_matmul_neg.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0439_abs_matmul_neg(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("abs_matmul_neg: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("abs_matmul_neg: device mismatch");
}
Tensor fused_0439_abs_matmul_neg(const Tensor& x,const Tensor& y) { validate_fused_0439_abs_matmul_neg(x,y); Tensor r=((x.matmul(y)).abs()).neg(); if(!r.defined()) throw std::runtime_error("abs_matmul_neg: undefined result"); return r; }
Tensor fused_0439_abs_matmul_neg_contiguous(const Tensor& x,const Tensor& y) { return fused_0439_abs_matmul_neg(x.contiguous(),y.contiguous()); }
Tensor fused_0439_abs_matmul_neg_cpu(const Tensor& x,const Tensor& y) { return fused_0439_abs_matmul_neg(x.cpu(),y.cpu()); }
Tensor fused_0439_abs_matmul_neg_clone(const Tensor& x,const Tensor& y) { return fused_0439_abs_matmul_neg(x.clone(),y.clone()); }
Tensor fused_0439_abs_matmul_neg_detached(const Tensor& x,const Tensor& y) { return fused_0439_abs_matmul_neg(x.detach(),y.detach()); }
bool fused_0439_abs_matmul_neg_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0439_abs_matmul_neg_numel(const Tensor& x) { return x.numel(); }
std::string fused_0439_abs_matmul_neg_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0439_abs_matmul_neg_name() { return "abs_matmul_neg"; }
}

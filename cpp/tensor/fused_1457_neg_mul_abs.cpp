#include "fused_1457_neg_mul_abs.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1457_neg_mul_abs(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("neg_mul_abs: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("neg_mul_abs: device mismatch");
}
Tensor fused_1457_neg_mul_abs(const Tensor& x,const Tensor& y) { validate_fused_1457_neg_mul_abs(x,y); Tensor r=((x * y).neg()).abs(); if(!r.defined()) throw std::runtime_error("neg_mul_abs: undefined result"); return r; }
Tensor fused_1457_neg_mul_abs_contiguous(const Tensor& x,const Tensor& y) { return fused_1457_neg_mul_abs(x.contiguous(),y.contiguous()); }
Tensor fused_1457_neg_mul_abs_cpu(const Tensor& x,const Tensor& y) { return fused_1457_neg_mul_abs(x.cpu(),y.cpu()); }
Tensor fused_1457_neg_mul_abs_clone(const Tensor& x,const Tensor& y) { return fused_1457_neg_mul_abs(x.clone(),y.clone()); }
Tensor fused_1457_neg_mul_abs_detached(const Tensor& x,const Tensor& y) { return fused_1457_neg_mul_abs(x.detach(),y.detach()); }
bool fused_1457_neg_mul_abs_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1457_neg_mul_abs_numel(const Tensor& x) { return x.numel(); }
std::string fused_1457_neg_mul_abs_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1457_neg_mul_abs_name() { return "neg_mul_abs"; }
}

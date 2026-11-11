#include "fused_1452_neg_mul_tanh.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1452_neg_mul_tanh(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("neg_mul_tanh: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("neg_mul_tanh: device mismatch");
}
Tensor fused_1452_neg_mul_tanh(const Tensor& x,const Tensor& y) { validate_fused_1452_neg_mul_tanh(x,y); Tensor r=((x * y).neg()).tanh(); if(!r.defined()) throw std::runtime_error("neg_mul_tanh: undefined result"); return r; }
Tensor fused_1452_neg_mul_tanh_contiguous(const Tensor& x,const Tensor& y) { return fused_1452_neg_mul_tanh(x.contiguous(),y.contiguous()); }
Tensor fused_1452_neg_mul_tanh_cpu(const Tensor& x,const Tensor& y) { return fused_1452_neg_mul_tanh(x.cpu(),y.cpu()); }
Tensor fused_1452_neg_mul_tanh_clone(const Tensor& x,const Tensor& y) { return fused_1452_neg_mul_tanh(x.clone(),y.clone()); }
Tensor fused_1452_neg_mul_tanh_detached(const Tensor& x,const Tensor& y) { return fused_1452_neg_mul_tanh(x.detach(),y.detach()); }
bool fused_1452_neg_mul_tanh_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1452_neg_mul_tanh_numel(const Tensor& x) { return x.numel(); }
std::string fused_1452_neg_mul_tanh_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1452_neg_mul_tanh_name() { return "neg_mul_tanh"; }
}

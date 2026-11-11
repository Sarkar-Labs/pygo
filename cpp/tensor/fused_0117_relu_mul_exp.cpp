#include "fused_0117_relu_mul_exp.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0117_relu_mul_exp(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("relu_mul_exp: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("relu_mul_exp: device mismatch");
}
Tensor fused_0117_relu_mul_exp(const Tensor& x,const Tensor& y) { validate_fused_0117_relu_mul_exp(x,y); Tensor r=((x * y).relu()).exp(); if(!r.defined()) throw std::runtime_error("relu_mul_exp: undefined result"); return r; }
Tensor fused_0117_relu_mul_exp_contiguous(const Tensor& x,const Tensor& y) { return fused_0117_relu_mul_exp(x.contiguous(),y.contiguous()); }
Tensor fused_0117_relu_mul_exp_cpu(const Tensor& x,const Tensor& y) { return fused_0117_relu_mul_exp(x.cpu(),y.cpu()); }
Tensor fused_0117_relu_mul_exp_clone(const Tensor& x,const Tensor& y) { return fused_0117_relu_mul_exp(x.clone(),y.clone()); }
Tensor fused_0117_relu_mul_exp_detached(const Tensor& x,const Tensor& y) { return fused_0117_relu_mul_exp(x.detach(),y.detach()); }
bool fused_0117_relu_mul_exp_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0117_relu_mul_exp_numel(const Tensor& x) { return x.numel(); }
std::string fused_0117_relu_mul_exp_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0117_relu_mul_exp_name() { return "relu_mul_exp"; }
}

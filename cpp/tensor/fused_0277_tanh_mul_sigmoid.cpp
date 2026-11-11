#include "fused_0277_tanh_mul_sigmoid.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0277_tanh_mul_sigmoid(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("tanh_mul_sigmoid: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("tanh_mul_sigmoid: device mismatch");
}
Tensor fused_0277_tanh_mul_sigmoid(const Tensor& x,const Tensor& y) { validate_fused_0277_tanh_mul_sigmoid(x,y); Tensor r=((x * y).tanh()).sigmoid(); if(!r.defined()) throw std::runtime_error("tanh_mul_sigmoid: undefined result"); return r; }
Tensor fused_0277_tanh_mul_sigmoid_contiguous(const Tensor& x,const Tensor& y) { return fused_0277_tanh_mul_sigmoid(x.contiguous(),y.contiguous()); }
Tensor fused_0277_tanh_mul_sigmoid_cpu(const Tensor& x,const Tensor& y) { return fused_0277_tanh_mul_sigmoid(x.cpu(),y.cpu()); }
Tensor fused_0277_tanh_mul_sigmoid_clone(const Tensor& x,const Tensor& y) { return fused_0277_tanh_mul_sigmoid(x.clone(),y.clone()); }
Tensor fused_0277_tanh_mul_sigmoid_detached(const Tensor& x,const Tensor& y) { return fused_0277_tanh_mul_sigmoid(x.detach(),y.detach()); }
bool fused_0277_tanh_mul_sigmoid_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0277_tanh_mul_sigmoid_numel(const Tensor& x) { return x.numel(); }
std::string fused_0277_tanh_mul_sigmoid_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0277_tanh_mul_sigmoid_name() { return "tanh_mul_sigmoid"; }
}

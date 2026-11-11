#include "fused_0352_tanh_mul_log2.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0352_tanh_mul_log2(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("tanh_mul_log2: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("tanh_mul_log2: device mismatch");
}
Tensor fused_0352_tanh_mul_log2(const Tensor& x,const Tensor& y) { validate_fused_0352_tanh_mul_log2(x,y); Tensor r=((x * y).tanh()).log2(); if(!r.defined()) throw std::runtime_error("tanh_mul_log2: undefined result"); return r; }
Tensor fused_0352_tanh_mul_log2_contiguous(const Tensor& x,const Tensor& y) { return fused_0352_tanh_mul_log2(x.contiguous(),y.contiguous()); }
Tensor fused_0352_tanh_mul_log2_cpu(const Tensor& x,const Tensor& y) { return fused_0352_tanh_mul_log2(x.cpu(),y.cpu()); }
Tensor fused_0352_tanh_mul_log2_clone(const Tensor& x,const Tensor& y) { return fused_0352_tanh_mul_log2(x.clone(),y.clone()); }
Tensor fused_0352_tanh_mul_log2_detached(const Tensor& x,const Tensor& y) { return fused_0352_tanh_mul_log2(x.detach(),y.detach()); }
bool fused_0352_tanh_mul_log2_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0352_tanh_mul_log2_numel(const Tensor& x) { return x.numel(); }
std::string fused_0352_tanh_mul_log2_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0352_tanh_mul_log2_name() { return "tanh_mul_log2"; }
}

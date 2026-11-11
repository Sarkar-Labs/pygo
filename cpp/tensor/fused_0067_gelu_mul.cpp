#include "fused_0067_gelu_mul.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0067_gelu_mul(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("gelu_mul: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("gelu_mul: device mismatch");
}
Tensor fused_0067_gelu_mul(const Tensor& x,const Tensor& y) { validate_fused_0067_gelu_mul(x,y); Tensor r=(x * y).gelu(); if(!r.defined()) throw std::runtime_error("gelu_mul: undefined result"); return r; }
Tensor fused_0067_gelu_mul_contiguous(const Tensor& x,const Tensor& y) { return fused_0067_gelu_mul(x.contiguous(),y.contiguous()); }
Tensor fused_0067_gelu_mul_cpu(const Tensor& x,const Tensor& y) { return fused_0067_gelu_mul(x.cpu(),y.cpu()); }
Tensor fused_0067_gelu_mul_clone(const Tensor& x,const Tensor& y) { return fused_0067_gelu_mul(x.clone(),y.clone()); }
Tensor fused_0067_gelu_mul_detached(const Tensor& x,const Tensor& y) { return fused_0067_gelu_mul(x.detach(),y.detach()); }
bool fused_0067_gelu_mul_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0067_gelu_mul_numel(const Tensor& x) { return x.numel(); }
std::string fused_0067_gelu_mul_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0067_gelu_mul_name() { return "gelu_mul"; }
}

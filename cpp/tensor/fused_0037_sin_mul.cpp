#include "fused_0037_sin_mul.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0037_sin_mul(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sin_mul: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sin_mul: device mismatch");
}
Tensor fused_0037_sin_mul(const Tensor& x,const Tensor& y) { validate_fused_0037_sin_mul(x,y); Tensor r=(x * y).sin(); if(!r.defined()) throw std::runtime_error("sin_mul: undefined result"); return r; }
Tensor fused_0037_sin_mul_contiguous(const Tensor& x,const Tensor& y) { return fused_0037_sin_mul(x.contiguous(),y.contiguous()); }
Tensor fused_0037_sin_mul_cpu(const Tensor& x,const Tensor& y) { return fused_0037_sin_mul(x.cpu(),y.cpu()); }
Tensor fused_0037_sin_mul_clone(const Tensor& x,const Tensor& y) { return fused_0037_sin_mul(x.clone(),y.clone()); }
Tensor fused_0037_sin_mul_detached(const Tensor& x,const Tensor& y) { return fused_0037_sin_mul(x.detach(),y.detach()); }
bool fused_0037_sin_mul_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0037_sin_mul_numel(const Tensor& x) { return x.numel(); }
std::string fused_0037_sin_mul_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0037_sin_mul_name() { return "sin_mul"; }
}

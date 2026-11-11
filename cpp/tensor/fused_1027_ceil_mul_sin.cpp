#include "fused_1027_ceil_mul_sin.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1027_ceil_mul_sin(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("ceil_mul_sin: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("ceil_mul_sin: device mismatch");
}
Tensor fused_1027_ceil_mul_sin(const Tensor& x,const Tensor& y) { validate_fused_1027_ceil_mul_sin(x,y); Tensor r=((x * y).ceil()).sin(); if(!r.defined()) throw std::runtime_error("ceil_mul_sin: undefined result"); return r; }
Tensor fused_1027_ceil_mul_sin_contiguous(const Tensor& x,const Tensor& y) { return fused_1027_ceil_mul_sin(x.contiguous(),y.contiguous()); }
Tensor fused_1027_ceil_mul_sin_cpu(const Tensor& x,const Tensor& y) { return fused_1027_ceil_mul_sin(x.cpu(),y.cpu()); }
Tensor fused_1027_ceil_mul_sin_clone(const Tensor& x,const Tensor& y) { return fused_1027_ceil_mul_sin(x.clone(),y.clone()); }
Tensor fused_1027_ceil_mul_sin_detached(const Tensor& x,const Tensor& y) { return fused_1027_ceil_mul_sin(x.detach(),y.detach()); }
bool fused_1027_ceil_mul_sin_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1027_ceil_mul_sin_numel(const Tensor& x) { return x.numel(); }
std::string fused_1027_ceil_mul_sin_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1027_ceil_mul_sin_name() { return "ceil_mul_sin"; }
}

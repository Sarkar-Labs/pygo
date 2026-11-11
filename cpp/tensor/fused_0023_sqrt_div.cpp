#include "fused_0023_sqrt_div.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0023_sqrt_div(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sqrt_div: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sqrt_div: device mismatch");
}
Tensor fused_0023_sqrt_div(const Tensor& x,const Tensor& y) { validate_fused_0023_sqrt_div(x,y); Tensor r=(x / y).sqrt(); if(!r.defined()) throw std::runtime_error("sqrt_div: undefined result"); return r; }
Tensor fused_0023_sqrt_div_contiguous(const Tensor& x,const Tensor& y) { return fused_0023_sqrt_div(x.contiguous(),y.contiguous()); }
Tensor fused_0023_sqrt_div_cpu(const Tensor& x,const Tensor& y) { return fused_0023_sqrt_div(x.cpu(),y.cpu()); }
Tensor fused_0023_sqrt_div_clone(const Tensor& x,const Tensor& y) { return fused_0023_sqrt_div(x.clone(),y.clone()); }
Tensor fused_0023_sqrt_div_detached(const Tensor& x,const Tensor& y) { return fused_0023_sqrt_div(x.detach(),y.detach()); }
bool fused_0023_sqrt_div_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0023_sqrt_div_numel(const Tensor& x) { return x.numel(); }
std::string fused_0023_sqrt_div_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0023_sqrt_div_name() { return "sqrt_div"; }
}

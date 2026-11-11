#include "fused_0038_sin_div.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0038_sin_div(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sin_div: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sin_div: device mismatch");
}
Tensor fused_0038_sin_div(const Tensor& x,const Tensor& y) { validate_fused_0038_sin_div(x,y); Tensor r=(x / y).sin(); if(!r.defined()) throw std::runtime_error("sin_div: undefined result"); return r; }
Tensor fused_0038_sin_div_contiguous(const Tensor& x,const Tensor& y) { return fused_0038_sin_div(x.contiguous(),y.contiguous()); }
Tensor fused_0038_sin_div_cpu(const Tensor& x,const Tensor& y) { return fused_0038_sin_div(x.cpu(),y.cpu()); }
Tensor fused_0038_sin_div_clone(const Tensor& x,const Tensor& y) { return fused_0038_sin_div(x.clone(),y.clone()); }
Tensor fused_0038_sin_div_detached(const Tensor& x,const Tensor& y) { return fused_0038_sin_div(x.detach(),y.detach()); }
bool fused_0038_sin_div_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0038_sin_div_numel(const Tensor& x) { return x.numel(); }
std::string fused_0038_sin_div_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0038_sin_div_name() { return "sin_div"; }
}

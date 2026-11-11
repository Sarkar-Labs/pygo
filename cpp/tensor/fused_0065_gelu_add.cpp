#include "fused_0065_gelu_add.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0065_gelu_add(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("gelu_add: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("gelu_add: device mismatch");
}
Tensor fused_0065_gelu_add(const Tensor& x,const Tensor& y) { validate_fused_0065_gelu_add(x,y); Tensor r=(x + y).gelu(); if(!r.defined()) throw std::runtime_error("gelu_add: undefined result"); return r; }
Tensor fused_0065_gelu_add_contiguous(const Tensor& x,const Tensor& y) { return fused_0065_gelu_add(x.contiguous(),y.contiguous()); }
Tensor fused_0065_gelu_add_cpu(const Tensor& x,const Tensor& y) { return fused_0065_gelu_add(x.cpu(),y.cpu()); }
Tensor fused_0065_gelu_add_clone(const Tensor& x,const Tensor& y) { return fused_0065_gelu_add(x.clone(),y.clone()); }
Tensor fused_0065_gelu_add_detached(const Tensor& x,const Tensor& y) { return fused_0065_gelu_add(x.detach(),y.detach()); }
bool fused_0065_gelu_add_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0065_gelu_add_numel(const Tensor& x) { return x.numel(); }
std::string fused_0065_gelu_add_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0065_gelu_add_name() { return "gelu_add"; }
}

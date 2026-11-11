#include "fused_0290_tanh_add_sqrt.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0290_tanh_add_sqrt(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("tanh_add_sqrt: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("tanh_add_sqrt: device mismatch");
}
Tensor fused_0290_tanh_add_sqrt(const Tensor& x,const Tensor& y) { validate_fused_0290_tanh_add_sqrt(x,y); Tensor r=((x + y).tanh()).sqrt(); if(!r.defined()) throw std::runtime_error("tanh_add_sqrt: undefined result"); return r; }
Tensor fused_0290_tanh_add_sqrt_contiguous(const Tensor& x,const Tensor& y) { return fused_0290_tanh_add_sqrt(x.contiguous(),y.contiguous()); }
Tensor fused_0290_tanh_add_sqrt_cpu(const Tensor& x,const Tensor& y) { return fused_0290_tanh_add_sqrt(x.cpu(),y.cpu()); }
Tensor fused_0290_tanh_add_sqrt_clone(const Tensor& x,const Tensor& y) { return fused_0290_tanh_add_sqrt(x.clone(),y.clone()); }
Tensor fused_0290_tanh_add_sqrt_detached(const Tensor& x,const Tensor& y) { return fused_0290_tanh_add_sqrt(x.detach(),y.detach()); }
bool fused_0290_tanh_add_sqrt_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0290_tanh_add_sqrt_numel(const Tensor& x) { return x.numel(); }
std::string fused_0290_tanh_add_sqrt_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0290_tanh_add_sqrt_name() { return "tanh_add_sqrt"; }
}

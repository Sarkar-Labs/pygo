#include "fused_0450_sqrt_add_relu.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0450_sqrt_add_relu(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sqrt_add_relu: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sqrt_add_relu: device mismatch");
}
Tensor fused_0450_sqrt_add_relu(const Tensor& x,const Tensor& y) { validate_fused_0450_sqrt_add_relu(x,y); Tensor r=((x + y).sqrt()).relu(); if(!r.defined()) throw std::runtime_error("sqrt_add_relu: undefined result"); return r; }
Tensor fused_0450_sqrt_add_relu_contiguous(const Tensor& x,const Tensor& y) { return fused_0450_sqrt_add_relu(x.contiguous(),y.contiguous()); }
Tensor fused_0450_sqrt_add_relu_cpu(const Tensor& x,const Tensor& y) { return fused_0450_sqrt_add_relu(x.cpu(),y.cpu()); }
Tensor fused_0450_sqrt_add_relu_clone(const Tensor& x,const Tensor& y) { return fused_0450_sqrt_add_relu(x.clone(),y.clone()); }
Tensor fused_0450_sqrt_add_relu_detached(const Tensor& x,const Tensor& y) { return fused_0450_sqrt_add_relu(x.detach(),y.detach()); }
bool fused_0450_sqrt_add_relu_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0450_sqrt_add_relu_numel(const Tensor& x) { return x.numel(); }
std::string fused_0450_sqrt_add_relu_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0450_sqrt_add_relu_name() { return "sqrt_add_relu"; }
}

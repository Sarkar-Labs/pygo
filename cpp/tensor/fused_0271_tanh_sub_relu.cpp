#include "fused_0271_tanh_sub_relu.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0271_tanh_sub_relu(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("tanh_sub_relu: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("tanh_sub_relu: device mismatch");
}
Tensor fused_0271_tanh_sub_relu(const Tensor& x,const Tensor& y) { validate_fused_0271_tanh_sub_relu(x,y); Tensor r=((x - y).tanh()).relu(); if(!r.defined()) throw std::runtime_error("tanh_sub_relu: undefined result"); return r; }
Tensor fused_0271_tanh_sub_relu_contiguous(const Tensor& x,const Tensor& y) { return fused_0271_tanh_sub_relu(x.contiguous(),y.contiguous()); }
Tensor fused_0271_tanh_sub_relu_cpu(const Tensor& x,const Tensor& y) { return fused_0271_tanh_sub_relu(x.cpu(),y.cpu()); }
Tensor fused_0271_tanh_sub_relu_clone(const Tensor& x,const Tensor& y) { return fused_0271_tanh_sub_relu(x.clone(),y.clone()); }
Tensor fused_0271_tanh_sub_relu_detached(const Tensor& x,const Tensor& y) { return fused_0271_tanh_sub_relu(x.detach(),y.detach()); }
bool fused_0271_tanh_sub_relu_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0271_tanh_sub_relu_numel(const Tensor& x) { return x.numel(); }
std::string fused_0271_tanh_sub_relu_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0271_tanh_sub_relu_name() { return "tanh_sub_relu"; }
}

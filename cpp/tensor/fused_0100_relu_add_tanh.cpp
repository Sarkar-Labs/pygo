#include "fused_0100_relu_add_tanh.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0100_relu_add_tanh(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("relu_add_tanh: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("relu_add_tanh: device mismatch");
}
Tensor fused_0100_relu_add_tanh(const Tensor& x,const Tensor& y) { validate_fused_0100_relu_add_tanh(x,y); Tensor r=((x + y).relu()).tanh(); if(!r.defined()) throw std::runtime_error("relu_add_tanh: undefined result"); return r; }
Tensor fused_0100_relu_add_tanh_contiguous(const Tensor& x,const Tensor& y) { return fused_0100_relu_add_tanh(x.contiguous(),y.contiguous()); }
Tensor fused_0100_relu_add_tanh_cpu(const Tensor& x,const Tensor& y) { return fused_0100_relu_add_tanh(x.cpu(),y.cpu()); }
Tensor fused_0100_relu_add_tanh_clone(const Tensor& x,const Tensor& y) { return fused_0100_relu_add_tanh(x.clone(),y.clone()); }
Tensor fused_0100_relu_add_tanh_detached(const Tensor& x,const Tensor& y) { return fused_0100_relu_add_tanh(x.detach(),y.detach()); }
bool fused_0100_relu_add_tanh_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0100_relu_add_tanh_numel(const Tensor& x) { return x.numel(); }
std::string fused_0100_relu_add_tanh_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0100_relu_add_tanh_name() { return "relu_add_tanh"; }
}

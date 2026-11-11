#include "fused_0154_relu_matmul_sign.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0154_relu_matmul_sign(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("relu_matmul_sign: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("relu_matmul_sign: device mismatch");
}
Tensor fused_0154_relu_matmul_sign(const Tensor& x,const Tensor& y) { validate_fused_0154_relu_matmul_sign(x,y); Tensor r=((x.matmul(y)).relu()).sign(); if(!r.defined()) throw std::runtime_error("relu_matmul_sign: undefined result"); return r; }
Tensor fused_0154_relu_matmul_sign_contiguous(const Tensor& x,const Tensor& y) { return fused_0154_relu_matmul_sign(x.contiguous(),y.contiguous()); }
Tensor fused_0154_relu_matmul_sign_cpu(const Tensor& x,const Tensor& y) { return fused_0154_relu_matmul_sign(x.cpu(),y.cpu()); }
Tensor fused_0154_relu_matmul_sign_clone(const Tensor& x,const Tensor& y) { return fused_0154_relu_matmul_sign(x.clone(),y.clone()); }
Tensor fused_0154_relu_matmul_sign_detached(const Tensor& x,const Tensor& y) { return fused_0154_relu_matmul_sign(x.detach(),y.detach()); }
bool fused_0154_relu_matmul_sign_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0154_relu_matmul_sign_numel(const Tensor& x) { return x.numel(); }
std::string fused_0154_relu_matmul_sign_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0154_relu_matmul_sign_name() { return "relu_matmul_sign"; }
}

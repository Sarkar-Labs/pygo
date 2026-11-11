#include "fused_0004_relu_matmul.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0004_relu_matmul(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("relu_matmul: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("relu_matmul: device mismatch");
}
Tensor fused_0004_relu_matmul(const Tensor& x,const Tensor& y) { validate_fused_0004_relu_matmul(x,y); Tensor r=(x.matmul(y)).relu(); if(!r.defined()) throw std::runtime_error("relu_matmul: undefined result"); return r; }
Tensor fused_0004_relu_matmul_contiguous(const Tensor& x,const Tensor& y) { return fused_0004_relu_matmul(x.contiguous(),y.contiguous()); }
Tensor fused_0004_relu_matmul_cpu(const Tensor& x,const Tensor& y) { return fused_0004_relu_matmul(x.cpu(),y.cpu()); }
Tensor fused_0004_relu_matmul_clone(const Tensor& x,const Tensor& y) { return fused_0004_relu_matmul(x.clone(),y.clone()); }
Tensor fused_0004_relu_matmul_detached(const Tensor& x,const Tensor& y) { return fused_0004_relu_matmul(x.detach(),y.detach()); }
bool fused_0004_relu_matmul_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0004_relu_matmul_numel(const Tensor& x) { return x.numel(); }
std::string fused_0004_relu_matmul_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0004_relu_matmul_name() { return "relu_matmul"; }
}

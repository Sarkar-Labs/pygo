#include "fused_0464_sqrt_matmul_tanh.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0464_sqrt_matmul_tanh(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sqrt_matmul_tanh: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sqrt_matmul_tanh: device mismatch");
}
Tensor fused_0464_sqrt_matmul_tanh(const Tensor& x,const Tensor& y) { validate_fused_0464_sqrt_matmul_tanh(x,y); Tensor r=((x.matmul(y)).sqrt()).tanh(); if(!r.defined()) throw std::runtime_error("sqrt_matmul_tanh: undefined result"); return r; }
Tensor fused_0464_sqrt_matmul_tanh_contiguous(const Tensor& x,const Tensor& y) { return fused_0464_sqrt_matmul_tanh(x.contiguous(),y.contiguous()); }
Tensor fused_0464_sqrt_matmul_tanh_cpu(const Tensor& x,const Tensor& y) { return fused_0464_sqrt_matmul_tanh(x.cpu(),y.cpu()); }
Tensor fused_0464_sqrt_matmul_tanh_clone(const Tensor& x,const Tensor& y) { return fused_0464_sqrt_matmul_tanh(x.clone(),y.clone()); }
Tensor fused_0464_sqrt_matmul_tanh_detached(const Tensor& x,const Tensor& y) { return fused_0464_sqrt_matmul_tanh(x.detach(),y.detach()); }
bool fused_0464_sqrt_matmul_tanh_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0464_sqrt_matmul_tanh_numel(const Tensor& x) { return x.numel(); }
std::string fused_0464_sqrt_matmul_tanh_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0464_sqrt_matmul_tanh_name() { return "sqrt_matmul_tanh"; }
}

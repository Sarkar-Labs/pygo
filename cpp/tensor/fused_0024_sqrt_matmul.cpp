#include "fused_0024_sqrt_matmul.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0024_sqrt_matmul(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sqrt_matmul: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sqrt_matmul: device mismatch");
}
Tensor fused_0024_sqrt_matmul(const Tensor& x,const Tensor& y) { validate_fused_0024_sqrt_matmul(x,y); Tensor r=(x.matmul(y)).sqrt(); if(!r.defined()) throw std::runtime_error("sqrt_matmul: undefined result"); return r; }
Tensor fused_0024_sqrt_matmul_contiguous(const Tensor& x,const Tensor& y) { return fused_0024_sqrt_matmul(x.contiguous(),y.contiguous()); }
Tensor fused_0024_sqrt_matmul_cpu(const Tensor& x,const Tensor& y) { return fused_0024_sqrt_matmul(x.cpu(),y.cpu()); }
Tensor fused_0024_sqrt_matmul_clone(const Tensor& x,const Tensor& y) { return fused_0024_sqrt_matmul(x.clone(),y.clone()); }
Tensor fused_0024_sqrt_matmul_detached(const Tensor& x,const Tensor& y) { return fused_0024_sqrt_matmul(x.detach(),y.detach()); }
bool fused_0024_sqrt_matmul_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0024_sqrt_matmul_numel(const Tensor& x) { return x.numel(); }
std::string fused_0024_sqrt_matmul_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0024_sqrt_matmul_name() { return "sqrt_matmul"; }
}

#include "fused_0564_exp_matmul_sqrt.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0564_exp_matmul_sqrt(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("exp_matmul_sqrt: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("exp_matmul_sqrt: device mismatch");
}
Tensor fused_0564_exp_matmul_sqrt(const Tensor& x,const Tensor& y) { validate_fused_0564_exp_matmul_sqrt(x,y); Tensor r=((x.matmul(y)).exp()).sqrt(); if(!r.defined()) throw std::runtime_error("exp_matmul_sqrt: undefined result"); return r; }
Tensor fused_0564_exp_matmul_sqrt_contiguous(const Tensor& x,const Tensor& y) { return fused_0564_exp_matmul_sqrt(x.contiguous(),y.contiguous()); }
Tensor fused_0564_exp_matmul_sqrt_cpu(const Tensor& x,const Tensor& y) { return fused_0564_exp_matmul_sqrt(x.cpu(),y.cpu()); }
Tensor fused_0564_exp_matmul_sqrt_clone(const Tensor& x,const Tensor& y) { return fused_0564_exp_matmul_sqrt(x.clone(),y.clone()); }
Tensor fused_0564_exp_matmul_sqrt_detached(const Tensor& x,const Tensor& y) { return fused_0564_exp_matmul_sqrt(x.detach(),y.detach()); }
bool fused_0564_exp_matmul_sqrt_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0564_exp_matmul_sqrt_numel(const Tensor& x) { return x.numel(); }
std::string fused_0564_exp_matmul_sqrt_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0564_exp_matmul_sqrt_name() { return "exp_matmul_sqrt"; }
}

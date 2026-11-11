#include "fused_1199_sign_matmul_exp.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1199_sign_matmul_exp(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sign_matmul_exp: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sign_matmul_exp: device mismatch");
}
Tensor fused_1199_sign_matmul_exp(const Tensor& x,const Tensor& y) { validate_fused_1199_sign_matmul_exp(x,y); Tensor r=((x.matmul(y)).sign()).exp(); if(!r.defined()) throw std::runtime_error("sign_matmul_exp: undefined result"); return r; }
Tensor fused_1199_sign_matmul_exp_contiguous(const Tensor& x,const Tensor& y) { return fused_1199_sign_matmul_exp(x.contiguous(),y.contiguous()); }
Tensor fused_1199_sign_matmul_exp_cpu(const Tensor& x,const Tensor& y) { return fused_1199_sign_matmul_exp(x.cpu(),y.cpu()); }
Tensor fused_1199_sign_matmul_exp_clone(const Tensor& x,const Tensor& y) { return fused_1199_sign_matmul_exp(x.clone(),y.clone()); }
Tensor fused_1199_sign_matmul_exp_detached(const Tensor& x,const Tensor& y) { return fused_1199_sign_matmul_exp(x.detach(),y.detach()); }
bool fused_1199_sign_matmul_exp_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1199_sign_matmul_exp_numel(const Tensor& x) { return x.numel(); }
std::string fused_1199_sign_matmul_exp_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1199_sign_matmul_exp_name() { return "sign_matmul_exp"; }
}

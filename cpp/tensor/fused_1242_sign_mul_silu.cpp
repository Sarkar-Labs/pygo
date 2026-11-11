#include "fused_1242_sign_mul_silu.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1242_sign_mul_silu(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sign_mul_silu: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sign_mul_silu: device mismatch");
}
Tensor fused_1242_sign_mul_silu(const Tensor& x,const Tensor& y) { validate_fused_1242_sign_mul_silu(x,y); Tensor r=((x * y).sign()).silu(); if(!r.defined()) throw std::runtime_error("sign_mul_silu: undefined result"); return r; }
Tensor fused_1242_sign_mul_silu_contiguous(const Tensor& x,const Tensor& y) { return fused_1242_sign_mul_silu(x.contiguous(),y.contiguous()); }
Tensor fused_1242_sign_mul_silu_cpu(const Tensor& x,const Tensor& y) { return fused_1242_sign_mul_silu(x.cpu(),y.cpu()); }
Tensor fused_1242_sign_mul_silu_clone(const Tensor& x,const Tensor& y) { return fused_1242_sign_mul_silu(x.clone(),y.clone()); }
Tensor fused_1242_sign_mul_silu_detached(const Tensor& x,const Tensor& y) { return fused_1242_sign_mul_silu(x.detach(),y.detach()); }
bool fused_1242_sign_mul_silu_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1242_sign_mul_silu_numel(const Tensor& x) { return x.numel(); }
std::string fused_1242_sign_mul_silu_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1242_sign_mul_silu_name() { return "sign_mul_silu"; }
}

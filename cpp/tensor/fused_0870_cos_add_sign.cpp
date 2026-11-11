#include "fused_0870_cos_add_sign.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0870_cos_add_sign(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("cos_add_sign: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("cos_add_sign: device mismatch");
}
Tensor fused_0870_cos_add_sign(const Tensor& x,const Tensor& y) { validate_fused_0870_cos_add_sign(x,y); Tensor r=((x + y).cos()).sign(); if(!r.defined()) throw std::runtime_error("cos_add_sign: undefined result"); return r; }
Tensor fused_0870_cos_add_sign_contiguous(const Tensor& x,const Tensor& y) { return fused_0870_cos_add_sign(x.contiguous(),y.contiguous()); }
Tensor fused_0870_cos_add_sign_cpu(const Tensor& x,const Tensor& y) { return fused_0870_cos_add_sign(x.cpu(),y.cpu()); }
Tensor fused_0870_cos_add_sign_clone(const Tensor& x,const Tensor& y) { return fused_0870_cos_add_sign(x.clone(),y.clone()); }
Tensor fused_0870_cos_add_sign_detached(const Tensor& x,const Tensor& y) { return fused_0870_cos_add_sign(x.detach(),y.detach()); }
bool fused_0870_cos_add_sign_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0870_cos_add_sign_numel(const Tensor& x) { return x.numel(); }
std::string fused_0870_cos_add_sign_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0870_cos_add_sign_name() { return "cos_add_sign"; }
}

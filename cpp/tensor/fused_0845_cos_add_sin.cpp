#include "fused_0845_cos_add_sin.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0845_cos_add_sin(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("cos_add_sin: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("cos_add_sin: device mismatch");
}
Tensor fused_0845_cos_add_sin(const Tensor& x,const Tensor& y) { validate_fused_0845_cos_add_sin(x,y); Tensor r=((x + y).cos()).sin(); if(!r.defined()) throw std::runtime_error("cos_add_sin: undefined result"); return r; }
Tensor fused_0845_cos_add_sin_contiguous(const Tensor& x,const Tensor& y) { return fused_0845_cos_add_sin(x.contiguous(),y.contiguous()); }
Tensor fused_0845_cos_add_sin_cpu(const Tensor& x,const Tensor& y) { return fused_0845_cos_add_sin(x.cpu(),y.cpu()); }
Tensor fused_0845_cos_add_sin_clone(const Tensor& x,const Tensor& y) { return fused_0845_cos_add_sin(x.clone(),y.clone()); }
Tensor fused_0845_cos_add_sin_detached(const Tensor& x,const Tensor& y) { return fused_0845_cos_add_sin(x.detach(),y.detach()); }
bool fused_0845_cos_add_sin_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0845_cos_add_sin_numel(const Tensor& x) { return x.numel(); }
std::string fused_0845_cos_add_sin_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0845_cos_add_sin_name() { return "cos_add_sin"; }
}

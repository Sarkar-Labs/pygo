#include "fused_0041_cos_sub.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0041_cos_sub(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("cos_sub: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("cos_sub: device mismatch");
}
Tensor fused_0041_cos_sub(const Tensor& x,const Tensor& y) { validate_fused_0041_cos_sub(x,y); Tensor r=(x - y).cos(); if(!r.defined()) throw std::runtime_error("cos_sub: undefined result"); return r; }
Tensor fused_0041_cos_sub_contiguous(const Tensor& x,const Tensor& y) { return fused_0041_cos_sub(x.contiguous(),y.contiguous()); }
Tensor fused_0041_cos_sub_cpu(const Tensor& x,const Tensor& y) { return fused_0041_cos_sub(x.cpu(),y.cpu()); }
Tensor fused_0041_cos_sub_clone(const Tensor& x,const Tensor& y) { return fused_0041_cos_sub(x.clone(),y.clone()); }
Tensor fused_0041_cos_sub_detached(const Tensor& x,const Tensor& y) { return fused_0041_cos_sub(x.detach(),y.detach()); }
bool fused_0041_cos_sub_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0041_cos_sub_numel(const Tensor& x) { return x.numel(); }
std::string fused_0041_cos_sub_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0041_cos_sub_name() { return "cos_sub"; }
}

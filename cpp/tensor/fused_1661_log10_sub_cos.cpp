#include "fused_1661_log10_sub_cos.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1661_log10_sub_cos(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log10_sub_cos: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log10_sub_cos: device mismatch");
}
Tensor fused_1661_log10_sub_cos(const Tensor& x,const Tensor& y) { validate_fused_1661_log10_sub_cos(x,y); Tensor r=((x - y).log10()).cos(); if(!r.defined()) throw std::runtime_error("log10_sub_cos: undefined result"); return r; }
Tensor fused_1661_log10_sub_cos_contiguous(const Tensor& x,const Tensor& y) { return fused_1661_log10_sub_cos(x.contiguous(),y.contiguous()); }
Tensor fused_1661_log10_sub_cos_cpu(const Tensor& x,const Tensor& y) { return fused_1661_log10_sub_cos(x.cpu(),y.cpu()); }
Tensor fused_1661_log10_sub_cos_clone(const Tensor& x,const Tensor& y) { return fused_1661_log10_sub_cos(x.clone(),y.clone()); }
Tensor fused_1661_log10_sub_cos_detached(const Tensor& x,const Tensor& y) { return fused_1661_log10_sub_cos(x.detach(),y.detach()); }
bool fused_1661_log10_sub_cos_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1661_log10_sub_cos_numel(const Tensor& x) { return x.numel(); }
std::string fused_1661_log10_sub_cos_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1661_log10_sub_cos_name() { return "log10_sub_cos"; }
}

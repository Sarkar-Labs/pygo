#include "fused_1573_log2_div_cos.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1573_log2_div_cos(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log2_div_cos: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log2_div_cos: device mismatch");
}
Tensor fused_1573_log2_div_cos(const Tensor& x,const Tensor& y) { validate_fused_1573_log2_div_cos(x,y); Tensor r=((x / y).log2()).cos(); if(!r.defined()) throw std::runtime_error("log2_div_cos: undefined result"); return r; }
Tensor fused_1573_log2_div_cos_contiguous(const Tensor& x,const Tensor& y) { return fused_1573_log2_div_cos(x.contiguous(),y.contiguous()); }
Tensor fused_1573_log2_div_cos_cpu(const Tensor& x,const Tensor& y) { return fused_1573_log2_div_cos(x.cpu(),y.cpu()); }
Tensor fused_1573_log2_div_cos_clone(const Tensor& x,const Tensor& y) { return fused_1573_log2_div_cos(x.clone(),y.clone()); }
Tensor fused_1573_log2_div_cos_detached(const Tensor& x,const Tensor& y) { return fused_1573_log2_div_cos(x.detach(),y.detach()); }
bool fused_1573_log2_div_cos_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1573_log2_div_cos_numel(const Tensor& x) { return x.numel(); }
std::string fused_1573_log2_div_cos_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1573_log2_div_cos_name() { return "log2_div_cos"; }
}

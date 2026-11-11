#include "fused_1551_log2_sub_sqrt.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1551_log2_sub_sqrt(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log2_sub_sqrt: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log2_sub_sqrt: device mismatch");
}
Tensor fused_1551_log2_sub_sqrt(const Tensor& x,const Tensor& y) { validate_fused_1551_log2_sub_sqrt(x,y); Tensor r=((x - y).log2()).sqrt(); if(!r.defined()) throw std::runtime_error("log2_sub_sqrt: undefined result"); return r; }
Tensor fused_1551_log2_sub_sqrt_contiguous(const Tensor& x,const Tensor& y) { return fused_1551_log2_sub_sqrt(x.contiguous(),y.contiguous()); }
Tensor fused_1551_log2_sub_sqrt_cpu(const Tensor& x,const Tensor& y) { return fused_1551_log2_sub_sqrt(x.cpu(),y.cpu()); }
Tensor fused_1551_log2_sub_sqrt_clone(const Tensor& x,const Tensor& y) { return fused_1551_log2_sub_sqrt(x.clone(),y.clone()); }
Tensor fused_1551_log2_sub_sqrt_detached(const Tensor& x,const Tensor& y) { return fused_1551_log2_sub_sqrt(x.detach(),y.detach()); }
bool fused_1551_log2_sub_sqrt_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1551_log2_sub_sqrt_numel(const Tensor& x) { return x.numel(); }
std::string fused_1551_log2_sub_sqrt_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1551_log2_sub_sqrt_name() { return "log2_sub_sqrt"; }
}

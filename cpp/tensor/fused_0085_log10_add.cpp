#include "fused_0085_log10_add.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0085_log10_add(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log10_add: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log10_add: device mismatch");
}
Tensor fused_0085_log10_add(const Tensor& x,const Tensor& y) { validate_fused_0085_log10_add(x,y); Tensor r=(x + y).log10(); if(!r.defined()) throw std::runtime_error("log10_add: undefined result"); return r; }
Tensor fused_0085_log10_add_contiguous(const Tensor& x,const Tensor& y) { return fused_0085_log10_add(x.contiguous(),y.contiguous()); }
Tensor fused_0085_log10_add_cpu(const Tensor& x,const Tensor& y) { return fused_0085_log10_add(x.cpu(),y.cpu()); }
Tensor fused_0085_log10_add_clone(const Tensor& x,const Tensor& y) { return fused_0085_log10_add(x.clone(),y.clone()); }
Tensor fused_0085_log10_add_detached(const Tensor& x,const Tensor& y) { return fused_0085_log10_add(x.detach(),y.detach()); }
bool fused_0085_log10_add_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0085_log10_add_numel(const Tensor& x) { return x.numel(); }
std::string fused_0085_log10_add_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0085_log10_add_name() { return "log10_add"; }
}

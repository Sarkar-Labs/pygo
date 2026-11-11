#include "fused_1435_silu_add_log10.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1435_silu_add_log10(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("silu_add_log10: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("silu_add_log10: device mismatch");
}
Tensor fused_1435_silu_add_log10(const Tensor& x,const Tensor& y) { validate_fused_1435_silu_add_log10(x,y); Tensor r=((x + y).silu()).log10(); if(!r.defined()) throw std::runtime_error("silu_add_log10: undefined result"); return r; }
Tensor fused_1435_silu_add_log10_contiguous(const Tensor& x,const Tensor& y) { return fused_1435_silu_add_log10(x.contiguous(),y.contiguous()); }
Tensor fused_1435_silu_add_log10_cpu(const Tensor& x,const Tensor& y) { return fused_1435_silu_add_log10(x.cpu(),y.cpu()); }
Tensor fused_1435_silu_add_log10_clone(const Tensor& x,const Tensor& y) { return fused_1435_silu_add_log10(x.clone(),y.clone()); }
Tensor fused_1435_silu_add_log10_detached(const Tensor& x,const Tensor& y) { return fused_1435_silu_add_log10(x.detach(),y.detach()); }
bool fused_1435_silu_add_log10_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1435_silu_add_log10_numel(const Tensor& x) { return x.numel(); }
std::string fused_1435_silu_add_log10_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1435_silu_add_log10_name() { return "silu_add_log10"; }
}

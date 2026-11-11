#include "fused_1545_log2_add_abs.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1545_log2_add_abs(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log2_add_abs: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log2_add_abs: device mismatch");
}
Tensor fused_1545_log2_add_abs(const Tensor& x,const Tensor& y) { validate_fused_1545_log2_add_abs(x,y); Tensor r=((x + y).log2()).abs(); if(!r.defined()) throw std::runtime_error("log2_add_abs: undefined result"); return r; }
Tensor fused_1545_log2_add_abs_contiguous(const Tensor& x,const Tensor& y) { return fused_1545_log2_add_abs(x.contiguous(),y.contiguous()); }
Tensor fused_1545_log2_add_abs_cpu(const Tensor& x,const Tensor& y) { return fused_1545_log2_add_abs(x.cpu(),y.cpu()); }
Tensor fused_1545_log2_add_abs_clone(const Tensor& x,const Tensor& y) { return fused_1545_log2_add_abs(x.clone(),y.clone()); }
Tensor fused_1545_log2_add_abs_detached(const Tensor& x,const Tensor& y) { return fused_1545_log2_add_abs(x.detach(),y.detach()); }
bool fused_1545_log2_add_abs_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1545_log2_add_abs_numel(const Tensor& x) { return x.numel(); }
std::string fused_1545_log2_add_abs_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1545_log2_add_abs_name() { return "log2_add_abs"; }
}

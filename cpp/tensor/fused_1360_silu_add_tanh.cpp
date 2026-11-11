#include "fused_1360_silu_add_tanh.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1360_silu_add_tanh(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("silu_add_tanh: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("silu_add_tanh: device mismatch");
}
Tensor fused_1360_silu_add_tanh(const Tensor& x,const Tensor& y) { validate_fused_1360_silu_add_tanh(x,y); Tensor r=((x + y).silu()).tanh(); if(!r.defined()) throw std::runtime_error("silu_add_tanh: undefined result"); return r; }
Tensor fused_1360_silu_add_tanh_contiguous(const Tensor& x,const Tensor& y) { return fused_1360_silu_add_tanh(x.contiguous(),y.contiguous()); }
Tensor fused_1360_silu_add_tanh_cpu(const Tensor& x,const Tensor& y) { return fused_1360_silu_add_tanh(x.cpu(),y.cpu()); }
Tensor fused_1360_silu_add_tanh_clone(const Tensor& x,const Tensor& y) { return fused_1360_silu_add_tanh(x.clone(),y.clone()); }
Tensor fused_1360_silu_add_tanh_detached(const Tensor& x,const Tensor& y) { return fused_1360_silu_add_tanh(x.detach(),y.detach()); }
bool fused_1360_silu_add_tanh_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1360_silu_add_tanh_numel(const Tensor& x) { return x.numel(); }
std::string fused_1360_silu_add_tanh_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1360_silu_add_tanh_name() { return "silu_add_tanh"; }
}

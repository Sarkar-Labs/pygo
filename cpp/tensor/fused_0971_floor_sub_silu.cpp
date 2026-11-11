#include "fused_0971_floor_sub_silu.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0971_floor_sub_silu(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("floor_sub_silu: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("floor_sub_silu: device mismatch");
}
Tensor fused_0971_floor_sub_silu(const Tensor& x,const Tensor& y) { validate_fused_0971_floor_sub_silu(x,y); Tensor r=((x - y).floor()).silu(); if(!r.defined()) throw std::runtime_error("floor_sub_silu: undefined result"); return r; }
Tensor fused_0971_floor_sub_silu_contiguous(const Tensor& x,const Tensor& y) { return fused_0971_floor_sub_silu(x.contiguous(),y.contiguous()); }
Tensor fused_0971_floor_sub_silu_cpu(const Tensor& x,const Tensor& y) { return fused_0971_floor_sub_silu(x.cpu(),y.cpu()); }
Tensor fused_0971_floor_sub_silu_clone(const Tensor& x,const Tensor& y) { return fused_0971_floor_sub_silu(x.clone(),y.clone()); }
Tensor fused_0971_floor_sub_silu_detached(const Tensor& x,const Tensor& y) { return fused_0971_floor_sub_silu(x.detach(),y.detach()); }
bool fused_0971_floor_sub_silu_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0971_floor_sub_silu_numel(const Tensor& x) { return x.numel(); }
std::string fused_0971_floor_sub_silu_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0971_floor_sub_silu_name() { return "floor_sub_silu"; }
}

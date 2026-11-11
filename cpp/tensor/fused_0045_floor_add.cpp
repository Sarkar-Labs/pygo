#include "fused_0045_floor_add.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0045_floor_add(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("floor_add: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("floor_add: device mismatch");
}
Tensor fused_0045_floor_add(const Tensor& x,const Tensor& y) { validate_fused_0045_floor_add(x,y); Tensor r=(x + y).floor(); if(!r.defined()) throw std::runtime_error("floor_add: undefined result"); return r; }
Tensor fused_0045_floor_add_contiguous(const Tensor& x,const Tensor& y) { return fused_0045_floor_add(x.contiguous(),y.contiguous()); }
Tensor fused_0045_floor_add_cpu(const Tensor& x,const Tensor& y) { return fused_0045_floor_add(x.cpu(),y.cpu()); }
Tensor fused_0045_floor_add_clone(const Tensor& x,const Tensor& y) { return fused_0045_floor_add(x.clone(),y.clone()); }
Tensor fused_0045_floor_add_detached(const Tensor& x,const Tensor& y) { return fused_0045_floor_add(x.detach(),y.detach()); }
bool fused_0045_floor_add_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0045_floor_add_numel(const Tensor& x) { return x.numel(); }
std::string fused_0045_floor_add_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0045_floor_add_name() { return "floor_add"; }
}

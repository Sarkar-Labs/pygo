#include "fused_0941_floor_sub_cos.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0941_floor_sub_cos(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("floor_sub_cos: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("floor_sub_cos: device mismatch");
}
Tensor fused_0941_floor_sub_cos(const Tensor& x,const Tensor& y) { validate_fused_0941_floor_sub_cos(x,y); Tensor r=((x - y).floor()).cos(); if(!r.defined()) throw std::runtime_error("floor_sub_cos: undefined result"); return r; }
Tensor fused_0941_floor_sub_cos_contiguous(const Tensor& x,const Tensor& y) { return fused_0941_floor_sub_cos(x.contiguous(),y.contiguous()); }
Tensor fused_0941_floor_sub_cos_cpu(const Tensor& x,const Tensor& y) { return fused_0941_floor_sub_cos(x.cpu(),y.cpu()); }
Tensor fused_0941_floor_sub_cos_clone(const Tensor& x,const Tensor& y) { return fused_0941_floor_sub_cos(x.clone(),y.clone()); }
Tensor fused_0941_floor_sub_cos_detached(const Tensor& x,const Tensor& y) { return fused_0941_floor_sub_cos(x.detach(),y.detach()); }
bool fused_0941_floor_sub_cos_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0941_floor_sub_cos_numel(const Tensor& x) { return x.numel(); }
std::string fused_0941_floor_sub_cos_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0941_floor_sub_cos_name() { return "floor_sub_cos"; }
}

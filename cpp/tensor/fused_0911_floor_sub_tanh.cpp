#include "fused_0911_floor_sub_tanh.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0911_floor_sub_tanh(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("floor_sub_tanh: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("floor_sub_tanh: device mismatch");
}
Tensor fused_0911_floor_sub_tanh(const Tensor& x,const Tensor& y) { validate_fused_0911_floor_sub_tanh(x,y); Tensor r=((x - y).floor()).tanh(); if(!r.defined()) throw std::runtime_error("floor_sub_tanh: undefined result"); return r; }
Tensor fused_0911_floor_sub_tanh_contiguous(const Tensor& x,const Tensor& y) { return fused_0911_floor_sub_tanh(x.contiguous(),y.contiguous()); }
Tensor fused_0911_floor_sub_tanh_cpu(const Tensor& x,const Tensor& y) { return fused_0911_floor_sub_tanh(x.cpu(),y.cpu()); }
Tensor fused_0911_floor_sub_tanh_clone(const Tensor& x,const Tensor& y) { return fused_0911_floor_sub_tanh(x.clone(),y.clone()); }
Tensor fused_0911_floor_sub_tanh_detached(const Tensor& x,const Tensor& y) { return fused_0911_floor_sub_tanh(x.detach(),y.detach()); }
bool fused_0911_floor_sub_tanh_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0911_floor_sub_tanh_numel(const Tensor& x) { return x.numel(); }
std::string fused_0911_floor_sub_tanh_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0911_floor_sub_tanh_name() { return "floor_sub_tanh"; }
}

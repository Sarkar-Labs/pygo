#include "fused_1306_gelu_sub_floor.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1306_gelu_sub_floor(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("gelu_sub_floor: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("gelu_sub_floor: device mismatch");
}
Tensor fused_1306_gelu_sub_floor(const Tensor& x,const Tensor& y) { validate_fused_1306_gelu_sub_floor(x,y); Tensor r=((x - y).gelu()).floor(); if(!r.defined()) throw std::runtime_error("gelu_sub_floor: undefined result"); return r; }
Tensor fused_1306_gelu_sub_floor_contiguous(const Tensor& x,const Tensor& y) { return fused_1306_gelu_sub_floor(x.contiguous(),y.contiguous()); }
Tensor fused_1306_gelu_sub_floor_cpu(const Tensor& x,const Tensor& y) { return fused_1306_gelu_sub_floor(x.cpu(),y.cpu()); }
Tensor fused_1306_gelu_sub_floor_clone(const Tensor& x,const Tensor& y) { return fused_1306_gelu_sub_floor(x.clone(),y.clone()); }
Tensor fused_1306_gelu_sub_floor_detached(const Tensor& x,const Tensor& y) { return fused_1306_gelu_sub_floor(x.detach(),y.detach()); }
bool fused_1306_gelu_sub_floor_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1306_gelu_sub_floor_numel(const Tensor& x) { return x.numel(); }
std::string fused_1306_gelu_sub_floor_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1306_gelu_sub_floor_name() { return "gelu_sub_floor"; }
}

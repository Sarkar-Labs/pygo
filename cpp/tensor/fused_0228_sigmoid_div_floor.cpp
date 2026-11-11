#include "fused_0228_sigmoid_div_floor.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0228_sigmoid_div_floor(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sigmoid_div_floor: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sigmoid_div_floor: device mismatch");
}
Tensor fused_0228_sigmoid_div_floor(const Tensor& x,const Tensor& y) { validate_fused_0228_sigmoid_div_floor(x,y); Tensor r=((x / y).sigmoid()).floor(); if(!r.defined()) throw std::runtime_error("sigmoid_div_floor: undefined result"); return r; }
Tensor fused_0228_sigmoid_div_floor_contiguous(const Tensor& x,const Tensor& y) { return fused_0228_sigmoid_div_floor(x.contiguous(),y.contiguous()); }
Tensor fused_0228_sigmoid_div_floor_cpu(const Tensor& x,const Tensor& y) { return fused_0228_sigmoid_div_floor(x.cpu(),y.cpu()); }
Tensor fused_0228_sigmoid_div_floor_clone(const Tensor& x,const Tensor& y) { return fused_0228_sigmoid_div_floor(x.clone(),y.clone()); }
Tensor fused_0228_sigmoid_div_floor_detached(const Tensor& x,const Tensor& y) { return fused_0228_sigmoid_div_floor(x.detach(),y.detach()); }
bool fused_0228_sigmoid_div_floor_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0228_sigmoid_div_floor_numel(const Tensor& x) { return x.numel(); }
std::string fused_0228_sigmoid_div_floor_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0228_sigmoid_div_floor_name() { return "sigmoid_div_floor"; }
}

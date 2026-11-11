#include "fused_0238_sigmoid_div_round.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0238_sigmoid_div_round(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sigmoid_div_round: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sigmoid_div_round: device mismatch");
}
Tensor fused_0238_sigmoid_div_round(const Tensor& x,const Tensor& y) { validate_fused_0238_sigmoid_div_round(x,y); Tensor r=((x / y).sigmoid()).round(); if(!r.defined()) throw std::runtime_error("sigmoid_div_round: undefined result"); return r; }
Tensor fused_0238_sigmoid_div_round_contiguous(const Tensor& x,const Tensor& y) { return fused_0238_sigmoid_div_round(x.contiguous(),y.contiguous()); }
Tensor fused_0238_sigmoid_div_round_cpu(const Tensor& x,const Tensor& y) { return fused_0238_sigmoid_div_round(x.cpu(),y.cpu()); }
Tensor fused_0238_sigmoid_div_round_clone(const Tensor& x,const Tensor& y) { return fused_0238_sigmoid_div_round(x.clone(),y.clone()); }
Tensor fused_0238_sigmoid_div_round_detached(const Tensor& x,const Tensor& y) { return fused_0238_sigmoid_div_round(x.detach(),y.detach()); }
bool fused_0238_sigmoid_div_round_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0238_sigmoid_div_round_numel(const Tensor& x) { return x.numel(); }
std::string fused_0238_sigmoid_div_round_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0238_sigmoid_div_round_name() { return "sigmoid_div_round"; }
}

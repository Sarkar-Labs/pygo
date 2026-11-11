#include "fused_1317_gelu_mul_round.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1317_gelu_mul_round(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("gelu_mul_round: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("gelu_mul_round: device mismatch");
}
Tensor fused_1317_gelu_mul_round(const Tensor& x,const Tensor& y) { validate_fused_1317_gelu_mul_round(x,y); Tensor r=((x * y).gelu()).round(); if(!r.defined()) throw std::runtime_error("gelu_mul_round: undefined result"); return r; }
Tensor fused_1317_gelu_mul_round_contiguous(const Tensor& x,const Tensor& y) { return fused_1317_gelu_mul_round(x.contiguous(),y.contiguous()); }
Tensor fused_1317_gelu_mul_round_cpu(const Tensor& x,const Tensor& y) { return fused_1317_gelu_mul_round(x.cpu(),y.cpu()); }
Tensor fused_1317_gelu_mul_round_clone(const Tensor& x,const Tensor& y) { return fused_1317_gelu_mul_round(x.clone(),y.clone()); }
Tensor fused_1317_gelu_mul_round_detached(const Tensor& x,const Tensor& y) { return fused_1317_gelu_mul_round(x.detach(),y.detach()); }
bool fused_1317_gelu_mul_round_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1317_gelu_mul_round_numel(const Tensor& x) { return x.numel(); }
std::string fused_1317_gelu_mul_round_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1317_gelu_mul_round_name() { return "gelu_mul_round"; }
}

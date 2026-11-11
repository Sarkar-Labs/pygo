#include "fused_1145_round_add_gelu.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1145_round_add_gelu(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("round_add_gelu: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("round_add_gelu: device mismatch");
}
Tensor fused_1145_round_add_gelu(const Tensor& x,const Tensor& y) { validate_fused_1145_round_add_gelu(x,y); Tensor r=((x + y).round()).gelu(); if(!r.defined()) throw std::runtime_error("round_add_gelu: undefined result"); return r; }
Tensor fused_1145_round_add_gelu_contiguous(const Tensor& x,const Tensor& y) { return fused_1145_round_add_gelu(x.contiguous(),y.contiguous()); }
Tensor fused_1145_round_add_gelu_cpu(const Tensor& x,const Tensor& y) { return fused_1145_round_add_gelu(x.cpu(),y.cpu()); }
Tensor fused_1145_round_add_gelu_clone(const Tensor& x,const Tensor& y) { return fused_1145_round_add_gelu(x.clone(),y.clone()); }
Tensor fused_1145_round_add_gelu_detached(const Tensor& x,const Tensor& y) { return fused_1145_round_add_gelu(x.detach(),y.detach()); }
bool fused_1145_round_add_gelu_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1145_round_add_gelu_numel(const Tensor& x) { return x.numel(); }
std::string fused_1145_round_add_gelu_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1145_round_add_gelu_name() { return "round_add_gelu"; }
}

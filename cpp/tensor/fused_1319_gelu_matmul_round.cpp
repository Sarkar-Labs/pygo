#include "fused_1319_gelu_matmul_round.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1319_gelu_matmul_round(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("gelu_matmul_round: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("gelu_matmul_round: device mismatch");
}
Tensor fused_1319_gelu_matmul_round(const Tensor& x,const Tensor& y) { validate_fused_1319_gelu_matmul_round(x,y); Tensor r=((x.matmul(y)).gelu()).round(); if(!r.defined()) throw std::runtime_error("gelu_matmul_round: undefined result"); return r; }
Tensor fused_1319_gelu_matmul_round_contiguous(const Tensor& x,const Tensor& y) { return fused_1319_gelu_matmul_round(x.contiguous(),y.contiguous()); }
Tensor fused_1319_gelu_matmul_round_cpu(const Tensor& x,const Tensor& y) { return fused_1319_gelu_matmul_round(x.cpu(),y.cpu()); }
Tensor fused_1319_gelu_matmul_round_clone(const Tensor& x,const Tensor& y) { return fused_1319_gelu_matmul_round(x.clone(),y.clone()); }
Tensor fused_1319_gelu_matmul_round_detached(const Tensor& x,const Tensor& y) { return fused_1319_gelu_matmul_round(x.detach(),y.detach()); }
bool fused_1319_gelu_matmul_round_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1319_gelu_matmul_round_numel(const Tensor& x) { return x.numel(); }
std::string fused_1319_gelu_matmul_round_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1319_gelu_matmul_round_name() { return "gelu_matmul_round"; }
}

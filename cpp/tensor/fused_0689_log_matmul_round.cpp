#include "fused_0689_log_matmul_round.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0689_log_matmul_round(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log_matmul_round: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log_matmul_round: device mismatch");
}
Tensor fused_0689_log_matmul_round(const Tensor& x,const Tensor& y) { validate_fused_0689_log_matmul_round(x,y); Tensor r=((x.matmul(y)).log()).round(); if(!r.defined()) throw std::runtime_error("log_matmul_round: undefined result"); return r; }
Tensor fused_0689_log_matmul_round_contiguous(const Tensor& x,const Tensor& y) { return fused_0689_log_matmul_round(x.contiguous(),y.contiguous()); }
Tensor fused_0689_log_matmul_round_cpu(const Tensor& x,const Tensor& y) { return fused_0689_log_matmul_round(x.cpu(),y.cpu()); }
Tensor fused_0689_log_matmul_round_clone(const Tensor& x,const Tensor& y) { return fused_0689_log_matmul_round(x.clone(),y.clone()); }
Tensor fused_0689_log_matmul_round_detached(const Tensor& x,const Tensor& y) { return fused_0689_log_matmul_round(x.detach(),y.detach()); }
bool fused_0689_log_matmul_round_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0689_log_matmul_round_numel(const Tensor& x) { return x.numel(); }
std::string fused_0689_log_matmul_round_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0689_log_matmul_round_name() { return "log_matmul_round"; }
}

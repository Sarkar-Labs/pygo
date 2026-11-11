#include "fused_1654_log10_matmul_log.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1654_log10_matmul_log(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log10_matmul_log: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log10_matmul_log: device mismatch");
}
Tensor fused_1654_log10_matmul_log(const Tensor& x,const Tensor& y) { validate_fused_1654_log10_matmul_log(x,y); Tensor r=((x.matmul(y)).log10()).log(); if(!r.defined()) throw std::runtime_error("log10_matmul_log: undefined result"); return r; }
Tensor fused_1654_log10_matmul_log_contiguous(const Tensor& x,const Tensor& y) { return fused_1654_log10_matmul_log(x.contiguous(),y.contiguous()); }
Tensor fused_1654_log10_matmul_log_cpu(const Tensor& x,const Tensor& y) { return fused_1654_log10_matmul_log(x.cpu(),y.cpu()); }
Tensor fused_1654_log10_matmul_log_clone(const Tensor& x,const Tensor& y) { return fused_1654_log10_matmul_log(x.clone(),y.clone()); }
Tensor fused_1654_log10_matmul_log_detached(const Tensor& x,const Tensor& y) { return fused_1654_log10_matmul_log(x.detach(),y.detach()); }
bool fused_1654_log10_matmul_log_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1654_log10_matmul_log_numel(const Tensor& x) { return x.numel(); }
std::string fused_1654_log10_matmul_log_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1654_log10_matmul_log_name() { return "log10_matmul_log"; }
}

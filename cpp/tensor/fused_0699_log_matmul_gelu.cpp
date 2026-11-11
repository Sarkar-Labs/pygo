#include "fused_0699_log_matmul_gelu.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0699_log_matmul_gelu(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log_matmul_gelu: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log_matmul_gelu: device mismatch");
}
Tensor fused_0699_log_matmul_gelu(const Tensor& x,const Tensor& y) { validate_fused_0699_log_matmul_gelu(x,y); Tensor r=((x.matmul(y)).log()).gelu(); if(!r.defined()) throw std::runtime_error("log_matmul_gelu: undefined result"); return r; }
Tensor fused_0699_log_matmul_gelu_contiguous(const Tensor& x,const Tensor& y) { return fused_0699_log_matmul_gelu(x.contiguous(),y.contiguous()); }
Tensor fused_0699_log_matmul_gelu_cpu(const Tensor& x,const Tensor& y) { return fused_0699_log_matmul_gelu(x.cpu(),y.cpu()); }
Tensor fused_0699_log_matmul_gelu_clone(const Tensor& x,const Tensor& y) { return fused_0699_log_matmul_gelu(x.clone(),y.clone()); }
Tensor fused_0699_log_matmul_gelu_detached(const Tensor& x,const Tensor& y) { return fused_0699_log_matmul_gelu(x.detach(),y.detach()); }
bool fused_0699_log_matmul_gelu_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0699_log_matmul_gelu_numel(const Tensor& x) { return x.numel(); }
std::string fused_0699_log_matmul_gelu_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0699_log_matmul_gelu_name() { return "log_matmul_gelu"; }
}

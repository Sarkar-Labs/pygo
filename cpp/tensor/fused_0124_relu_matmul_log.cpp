#include "fused_0124_relu_matmul_log.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0124_relu_matmul_log(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("relu_matmul_log: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("relu_matmul_log: device mismatch");
}
Tensor fused_0124_relu_matmul_log(const Tensor& x,const Tensor& y) { validate_fused_0124_relu_matmul_log(x,y); Tensor r=((x.matmul(y)).relu()).log(); if(!r.defined()) throw std::runtime_error("relu_matmul_log: undefined result"); return r; }
Tensor fused_0124_relu_matmul_log_contiguous(const Tensor& x,const Tensor& y) { return fused_0124_relu_matmul_log(x.contiguous(),y.contiguous()); }
Tensor fused_0124_relu_matmul_log_cpu(const Tensor& x,const Tensor& y) { return fused_0124_relu_matmul_log(x.cpu(),y.cpu()); }
Tensor fused_0124_relu_matmul_log_clone(const Tensor& x,const Tensor& y) { return fused_0124_relu_matmul_log(x.clone(),y.clone()); }
Tensor fused_0124_relu_matmul_log_detached(const Tensor& x,const Tensor& y) { return fused_0124_relu_matmul_log(x.detach(),y.detach()); }
bool fused_0124_relu_matmul_log_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0124_relu_matmul_log_numel(const Tensor& x) { return x.numel(); }
std::string fused_0124_relu_matmul_log_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0124_relu_matmul_log_name() { return "relu_matmul_log"; }
}

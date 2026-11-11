#include "fused_1544_log2_matmul_tanh.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1544_log2_matmul_tanh(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log2_matmul_tanh: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log2_matmul_tanh: device mismatch");
}
Tensor fused_1544_log2_matmul_tanh(const Tensor& x,const Tensor& y) { validate_fused_1544_log2_matmul_tanh(x,y); Tensor r=((x.matmul(y)).log2()).tanh(); if(!r.defined()) throw std::runtime_error("log2_matmul_tanh: undefined result"); return r; }
Tensor fused_1544_log2_matmul_tanh_contiguous(const Tensor& x,const Tensor& y) { return fused_1544_log2_matmul_tanh(x.contiguous(),y.contiguous()); }
Tensor fused_1544_log2_matmul_tanh_cpu(const Tensor& x,const Tensor& y) { return fused_1544_log2_matmul_tanh(x.cpu(),y.cpu()); }
Tensor fused_1544_log2_matmul_tanh_clone(const Tensor& x,const Tensor& y) { return fused_1544_log2_matmul_tanh(x.clone(),y.clone()); }
Tensor fused_1544_log2_matmul_tanh_detached(const Tensor& x,const Tensor& y) { return fused_1544_log2_matmul_tanh(x.detach(),y.detach()); }
bool fused_1544_log2_matmul_tanh_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1544_log2_matmul_tanh_numel(const Tensor& x) { return x.numel(); }
std::string fused_1544_log2_matmul_tanh_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1544_log2_matmul_tanh_name() { return "log2_matmul_tanh"; }
}

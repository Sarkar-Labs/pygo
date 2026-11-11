#include "fused_0642_log_mul_tanh.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0642_log_mul_tanh(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log_mul_tanh: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log_mul_tanh: device mismatch");
}
Tensor fused_0642_log_mul_tanh(const Tensor& x,const Tensor& y) { validate_fused_0642_log_mul_tanh(x,y); Tensor r=((x * y).log()).tanh(); if(!r.defined()) throw std::runtime_error("log_mul_tanh: undefined result"); return r; }
Tensor fused_0642_log_mul_tanh_contiguous(const Tensor& x,const Tensor& y) { return fused_0642_log_mul_tanh(x.contiguous(),y.contiguous()); }
Tensor fused_0642_log_mul_tanh_cpu(const Tensor& x,const Tensor& y) { return fused_0642_log_mul_tanh(x.cpu(),y.cpu()); }
Tensor fused_0642_log_mul_tanh_clone(const Tensor& x,const Tensor& y) { return fused_0642_log_mul_tanh(x.clone(),y.clone()); }
Tensor fused_0642_log_mul_tanh_detached(const Tensor& x,const Tensor& y) { return fused_0642_log_mul_tanh(x.detach(),y.detach()); }
bool fused_0642_log_mul_tanh_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0642_log_mul_tanh_numel(const Tensor& x) { return x.numel(); }
std::string fused_0642_log_mul_tanh_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0642_log_mul_tanh_name() { return "log_mul_tanh"; }
}

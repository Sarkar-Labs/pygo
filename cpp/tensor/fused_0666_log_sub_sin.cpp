#include "fused_0666_log_sub_sin.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0666_log_sub_sin(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log_sub_sin: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log_sub_sin: device mismatch");
}
Tensor fused_0666_log_sub_sin(const Tensor& x,const Tensor& y) { validate_fused_0666_log_sub_sin(x,y); Tensor r=((x - y).log()).sin(); if(!r.defined()) throw std::runtime_error("log_sub_sin: undefined result"); return r; }
Tensor fused_0666_log_sub_sin_contiguous(const Tensor& x,const Tensor& y) { return fused_0666_log_sub_sin(x.contiguous(),y.contiguous()); }
Tensor fused_0666_log_sub_sin_cpu(const Tensor& x,const Tensor& y) { return fused_0666_log_sub_sin(x.cpu(),y.cpu()); }
Tensor fused_0666_log_sub_sin_clone(const Tensor& x,const Tensor& y) { return fused_0666_log_sub_sin(x.clone(),y.clone()); }
Tensor fused_0666_log_sub_sin_detached(const Tensor& x,const Tensor& y) { return fused_0666_log_sub_sin(x.detach(),y.detach()); }
bool fused_0666_log_sub_sin_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0666_log_sub_sin_numel(const Tensor& x) { return x.numel(); }
std::string fused_0666_log_sub_sin_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0666_log_sub_sin_name() { return "log_sub_sin"; }
}

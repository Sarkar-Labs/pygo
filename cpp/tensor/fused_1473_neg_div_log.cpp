#include "fused_1473_neg_div_log.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1473_neg_div_log(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("neg_div_log: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("neg_div_log: device mismatch");
}
Tensor fused_1473_neg_div_log(const Tensor& x,const Tensor& y) { validate_fused_1473_neg_div_log(x,y); Tensor r=((x / y).neg()).log(); if(!r.defined()) throw std::runtime_error("neg_div_log: undefined result"); return r; }
Tensor fused_1473_neg_div_log_contiguous(const Tensor& x,const Tensor& y) { return fused_1473_neg_div_log(x.contiguous(),y.contiguous()); }
Tensor fused_1473_neg_div_log_cpu(const Tensor& x,const Tensor& y) { return fused_1473_neg_div_log(x.cpu(),y.cpu()); }
Tensor fused_1473_neg_div_log_clone(const Tensor& x,const Tensor& y) { return fused_1473_neg_div_log(x.clone(),y.clone()); }
Tensor fused_1473_neg_div_log_detached(const Tensor& x,const Tensor& y) { return fused_1473_neg_div_log(x.detach(),y.detach()); }
bool fused_1473_neg_div_log_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1473_neg_div_log_numel(const Tensor& x) { return x.numel(); }
std::string fused_1473_neg_div_log_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1473_neg_div_log_name() { return "neg_div_log"; }
}

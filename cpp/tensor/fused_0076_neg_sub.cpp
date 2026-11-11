#include "fused_0076_neg_sub.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0076_neg_sub(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("neg_sub: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("neg_sub: device mismatch");
}
Tensor fused_0076_neg_sub(const Tensor& x,const Tensor& y) { validate_fused_0076_neg_sub(x,y); Tensor r=(x - y).neg(); if(!r.defined()) throw std::runtime_error("neg_sub: undefined result"); return r; }
Tensor fused_0076_neg_sub_contiguous(const Tensor& x,const Tensor& y) { return fused_0076_neg_sub(x.contiguous(),y.contiguous()); }
Tensor fused_0076_neg_sub_cpu(const Tensor& x,const Tensor& y) { return fused_0076_neg_sub(x.cpu(),y.cpu()); }
Tensor fused_0076_neg_sub_clone(const Tensor& x,const Tensor& y) { return fused_0076_neg_sub(x.clone(),y.clone()); }
Tensor fused_0076_neg_sub_detached(const Tensor& x,const Tensor& y) { return fused_0076_neg_sub(x.detach(),y.detach()); }
bool fused_0076_neg_sub_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0076_neg_sub_numel(const Tensor& x) { return x.numel(); }
std::string fused_0076_neg_sub_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0076_neg_sub_name() { return "neg_sub"; }
}

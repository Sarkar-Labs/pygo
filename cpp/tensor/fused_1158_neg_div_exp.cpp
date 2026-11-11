#include "fused_1158_neg_div_exp.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("neg_div_exp: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("neg_div_exp: device mismatch");
}
Tensor fused_1158_neg_div_exp(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x / y.neg()).exp();
    if (!r.defined()) throw std::runtime_error("neg_div_exp: undefined result");
    return r;
}
Tensor fused_1158_neg_div_exp_contiguous(const Tensor& x,const Tensor& y) { return fused_1158_neg_div_exp(x.contiguous(),y.contiguous()); }
Tensor fused_1158_neg_div_exp_cpu(const Tensor& x,const Tensor& y) { return fused_1158_neg_div_exp(x.cpu(),y.cpu()); }
Tensor fused_1158_neg_div_exp_clone(const Tensor& x,const Tensor& y) { return fused_1158_neg_div_exp(x.clone(),y.clone()); }
Tensor fused_1158_neg_div_exp_detached(const Tensor& x,const Tensor& y) { return fused_1158_neg_div_exp(x.detach(),y.detach()); }
bool fused_1158_neg_div_exp_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1158_neg_div_exp_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_1158_neg_div_exp_numel(const Tensor& x) { return x.numel(); }
DType fused_1158_neg_div_exp_dtype(const Tensor& x) { return x.dtype(); }
Device fused_1158_neg_div_exp_device(const Tensor& x) { return x.device(); }
std::string fused_1158_neg_div_exp_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1158_neg_div_exp_name() { return "neg_div_exp"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

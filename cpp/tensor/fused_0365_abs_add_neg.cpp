#include "fused_0365_abs_add_neg.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("abs_add_neg: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("abs_add_neg: device mismatch");
}
Tensor fused_0365_abs_add_neg(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x + y.abs()).neg();
    if (!r.defined()) throw std::runtime_error("abs_add_neg: undefined result");
    return r;
}
Tensor fused_0365_abs_add_neg_contiguous(const Tensor& x,const Tensor& y) { return fused_0365_abs_add_neg(x.contiguous(),y.contiguous()); }
Tensor fused_0365_abs_add_neg_cpu(const Tensor& x,const Tensor& y) { return fused_0365_abs_add_neg(x.cpu(),y.cpu()); }
Tensor fused_0365_abs_add_neg_clone(const Tensor& x,const Tensor& y) { return fused_0365_abs_add_neg(x.clone(),y.clone()); }
Tensor fused_0365_abs_add_neg_detached(const Tensor& x,const Tensor& y) { return fused_0365_abs_add_neg(x.detach(),y.detach()); }
bool fused_0365_abs_add_neg_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0365_abs_add_neg_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0365_abs_add_neg_numel(const Tensor& x) { return x.numel(); }
DType fused_0365_abs_add_neg_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0365_abs_add_neg_device(const Tensor& x) { return x.device(); }
std::string fused_0365_abs_add_neg_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0365_abs_add_neg_name() { return "abs_add_neg"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

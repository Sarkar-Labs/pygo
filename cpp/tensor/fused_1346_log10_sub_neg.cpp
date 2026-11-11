#include "fused_1346_log10_sub_neg.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log10_sub_neg: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log10_sub_neg: device mismatch");
}
Tensor fused_1346_log10_sub_neg(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x - y.log10()).neg();
    if (!r.defined()) throw std::runtime_error("log10_sub_neg: undefined result");
    return r;
}
Tensor fused_1346_log10_sub_neg_contiguous(const Tensor& x,const Tensor& y) { return fused_1346_log10_sub_neg(x.contiguous(),y.contiguous()); }
Tensor fused_1346_log10_sub_neg_cpu(const Tensor& x,const Tensor& y) { return fused_1346_log10_sub_neg(x.cpu(),y.cpu()); }
Tensor fused_1346_log10_sub_neg_clone(const Tensor& x,const Tensor& y) { return fused_1346_log10_sub_neg(x.clone(),y.clone()); }
Tensor fused_1346_log10_sub_neg_detached(const Tensor& x,const Tensor& y) { return fused_1346_log10_sub_neg(x.detach(),y.detach()); }
bool fused_1346_log10_sub_neg_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1346_log10_sub_neg_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_1346_log10_sub_neg_numel(const Tensor& x) { return x.numel(); }
DType fused_1346_log10_sub_neg_dtype(const Tensor& x) { return x.dtype(); }
Device fused_1346_log10_sub_neg_device(const Tensor& x) { return x.device(); }
std::string fused_1346_log10_sub_neg_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1346_log10_sub_neg_name() { return "log10_sub_neg"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

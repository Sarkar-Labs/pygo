#include "fused_1161_neg_sub_log.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("neg_sub_log: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("neg_sub_log: device mismatch");
}
Tensor fused_1161_neg_sub_log(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x - y.neg()).log();
    if (!r.defined()) throw std::runtime_error("neg_sub_log: undefined result");
    return r;
}
Tensor fused_1161_neg_sub_log_contiguous(const Tensor& x,const Tensor& y) { return fused_1161_neg_sub_log(x.contiguous(),y.contiguous()); }
Tensor fused_1161_neg_sub_log_cpu(const Tensor& x,const Tensor& y) { return fused_1161_neg_sub_log(x.cpu(),y.cpu()); }
Tensor fused_1161_neg_sub_log_clone(const Tensor& x,const Tensor& y) { return fused_1161_neg_sub_log(x.clone(),y.clone()); }
Tensor fused_1161_neg_sub_log_detached(const Tensor& x,const Tensor& y) { return fused_1161_neg_sub_log(x.detach(),y.detach()); }
bool fused_1161_neg_sub_log_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1161_neg_sub_log_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_1161_neg_sub_log_numel(const Tensor& x) { return x.numel(); }
DType fused_1161_neg_sub_log_dtype(const Tensor& x) { return x.dtype(); }
Device fused_1161_neg_sub_log_device(const Tensor& x) { return x.device(); }
std::string fused_1161_neg_sub_log_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1161_neg_sub_log_name() { return "neg_sub_log"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

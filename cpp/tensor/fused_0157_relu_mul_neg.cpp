#include "fused_0157_relu_mul_neg.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("relu_mul_neg: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("relu_mul_neg: device mismatch");
}
Tensor fused_0157_relu_mul_neg(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x * y.relu()).neg();
    if (!r.defined()) throw std::runtime_error("relu_mul_neg: undefined result");
    return r;
}
Tensor fused_0157_relu_mul_neg_contiguous(const Tensor& x,const Tensor& y) { return fused_0157_relu_mul_neg(x.contiguous(),y.contiguous()); }
Tensor fused_0157_relu_mul_neg_cpu(const Tensor& x,const Tensor& y) { return fused_0157_relu_mul_neg(x.cpu(),y.cpu()); }
Tensor fused_0157_relu_mul_neg_clone(const Tensor& x,const Tensor& y) { return fused_0157_relu_mul_neg(x.clone(),y.clone()); }
Tensor fused_0157_relu_mul_neg_detached(const Tensor& x,const Tensor& y) { return fused_0157_relu_mul_neg(x.detach(),y.detach()); }
bool fused_0157_relu_mul_neg_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0157_relu_mul_neg_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0157_relu_mul_neg_numel(const Tensor& x) { return x.numel(); }
DType fused_0157_relu_mul_neg_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0157_relu_mul_neg_device(const Tensor& x) { return x.device(); }
std::string fused_0157_relu_mul_neg_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0157_relu_mul_neg_name() { return "relu_mul_neg"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

#include "fused_0941_sign_sub_sqrt.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sign_sub_sqrt: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sign_sub_sqrt: device mismatch");
}
Tensor fused_0941_sign_sub_sqrt(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x - y.sign()).sqrt();
    if (!r.defined()) throw std::runtime_error("sign_sub_sqrt: undefined result");
    return r;
}
Tensor fused_0941_sign_sub_sqrt_contiguous(const Tensor& x,const Tensor& y) { return fused_0941_sign_sub_sqrt(x.contiguous(),y.contiguous()); }
Tensor fused_0941_sign_sub_sqrt_cpu(const Tensor& x,const Tensor& y) { return fused_0941_sign_sub_sqrt(x.cpu(),y.cpu()); }
Tensor fused_0941_sign_sub_sqrt_clone(const Tensor& x,const Tensor& y) { return fused_0941_sign_sub_sqrt(x.clone(),y.clone()); }
Tensor fused_0941_sign_sub_sqrt_detached(const Tensor& x,const Tensor& y) { return fused_0941_sign_sub_sqrt(x.detach(),y.detach()); }
bool fused_0941_sign_sub_sqrt_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0941_sign_sub_sqrt_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0941_sign_sub_sqrt_numel(const Tensor& x) { return x.numel(); }
DType fused_0941_sign_sub_sqrt_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0941_sign_sub_sqrt_device(const Tensor& x) { return x.device(); }
std::string fused_0941_sign_sub_sqrt_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0941_sign_sub_sqrt_name() { return "sign_sub_sqrt"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

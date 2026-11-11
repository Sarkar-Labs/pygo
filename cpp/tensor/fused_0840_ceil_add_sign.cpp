#include "fused_0840_ceil_add_sign.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("ceil_add_sign: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("ceil_add_sign: device mismatch");
}
Tensor fused_0840_ceil_add_sign(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x + y.ceil()).sign();
    if (!r.defined()) throw std::runtime_error("ceil_add_sign: undefined result");
    return r;
}
Tensor fused_0840_ceil_add_sign_contiguous(const Tensor& x,const Tensor& y) { return fused_0840_ceil_add_sign(x.contiguous(),y.contiguous()); }
Tensor fused_0840_ceil_add_sign_cpu(const Tensor& x,const Tensor& y) { return fused_0840_ceil_add_sign(x.cpu(),y.cpu()); }
Tensor fused_0840_ceil_add_sign_clone(const Tensor& x,const Tensor& y) { return fused_0840_ceil_add_sign(x.clone(),y.clone()); }
Tensor fused_0840_ceil_add_sign_detached(const Tensor& x,const Tensor& y) { return fused_0840_ceil_add_sign(x.detach(),y.detach()); }
bool fused_0840_ceil_add_sign_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0840_ceil_add_sign_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0840_ceil_add_sign_numel(const Tensor& x) { return x.numel(); }
DType fused_0840_ceil_add_sign_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0840_ceil_add_sign_device(const Tensor& x) { return x.device(); }
std::string fused_0840_ceil_add_sign_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0840_ceil_add_sign_name() { return "ceil_add_sign"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

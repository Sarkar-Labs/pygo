#include "fused_0976_sign_sub_round.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sign_sub_round: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sign_sub_round: device mismatch");
}
Tensor fused_0976_sign_sub_round(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x - y.sign()).round();
    if (!r.defined()) throw std::runtime_error("sign_sub_round: undefined result");
    return r;
}
Tensor fused_0976_sign_sub_round_contiguous(const Tensor& x,const Tensor& y) { return fused_0976_sign_sub_round(x.contiguous(),y.contiguous()); }
Tensor fused_0976_sign_sub_round_cpu(const Tensor& x,const Tensor& y) { return fused_0976_sign_sub_round(x.cpu(),y.cpu()); }
Tensor fused_0976_sign_sub_round_clone(const Tensor& x,const Tensor& y) { return fused_0976_sign_sub_round(x.clone(),y.clone()); }
Tensor fused_0976_sign_sub_round_detached(const Tensor& x,const Tensor& y) { return fused_0976_sign_sub_round(x.detach(),y.detach()); }
bool fused_0976_sign_sub_round_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0976_sign_sub_round_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0976_sign_sub_round_numel(const Tensor& x) { return x.numel(); }
DType fused_0976_sign_sub_round_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0976_sign_sub_round_device(const Tensor& x) { return x.device(); }
std::string fused_0976_sign_sub_round_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0976_sign_sub_round_name() { return "sign_sub_round"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

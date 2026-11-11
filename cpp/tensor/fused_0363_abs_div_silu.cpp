#include "fused_0363_abs_div_silu.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("abs_div_silu: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("abs_div_silu: device mismatch");
}
Tensor fused_0363_abs_div_silu(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x / y.abs()).silu();
    if (!r.defined()) throw std::runtime_error("abs_div_silu: undefined result");
    return r;
}
Tensor fused_0363_abs_div_silu_contiguous(const Tensor& x,const Tensor& y) { return fused_0363_abs_div_silu(x.contiguous(),y.contiguous()); }
Tensor fused_0363_abs_div_silu_cpu(const Tensor& x,const Tensor& y) { return fused_0363_abs_div_silu(x.cpu(),y.cpu()); }
Tensor fused_0363_abs_div_silu_clone(const Tensor& x,const Tensor& y) { return fused_0363_abs_div_silu(x.clone(),y.clone()); }
Tensor fused_0363_abs_div_silu_detached(const Tensor& x,const Tensor& y) { return fused_0363_abs_div_silu(x.detach(),y.detach()); }
bool fused_0363_abs_div_silu_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0363_abs_div_silu_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0363_abs_div_silu_numel(const Tensor& x) { return x.numel(); }
DType fused_0363_abs_div_silu_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0363_abs_div_silu_device(const Tensor& x) { return x.device(); }
std::string fused_0363_abs_div_silu_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0363_abs_div_silu_name() { return "abs_div_silu"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

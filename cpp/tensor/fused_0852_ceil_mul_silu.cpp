#include "fused_0852_ceil_mul_silu.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("ceil_mul_silu: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("ceil_mul_silu: device mismatch");
}
Tensor fused_0852_ceil_mul_silu(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x * y.ceil()).silu();
    if (!r.defined()) throw std::runtime_error("ceil_mul_silu: undefined result");
    return r;
}
Tensor fused_0852_ceil_mul_silu_contiguous(const Tensor& x,const Tensor& y) { return fused_0852_ceil_mul_silu(x.contiguous(),y.contiguous()); }
Tensor fused_0852_ceil_mul_silu_cpu(const Tensor& x,const Tensor& y) { return fused_0852_ceil_mul_silu(x.cpu(),y.cpu()); }
Tensor fused_0852_ceil_mul_silu_clone(const Tensor& x,const Tensor& y) { return fused_0852_ceil_mul_silu(x.clone(),y.clone()); }
Tensor fused_0852_ceil_mul_silu_detached(const Tensor& x,const Tensor& y) { return fused_0852_ceil_mul_silu(x.detach(),y.detach()); }
bool fused_0852_ceil_mul_silu_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0852_ceil_mul_silu_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0852_ceil_mul_silu_numel(const Tensor& x) { return x.numel(); }
DType fused_0852_ceil_mul_silu_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0852_ceil_mul_silu_device(const Tensor& x) { return x.device(); }
std::string fused_0852_ceil_mul_silu_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0852_ceil_mul_silu_name() { return "ceil_mul_silu"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

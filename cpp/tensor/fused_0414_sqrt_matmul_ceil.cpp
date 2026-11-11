#include "fused_0414_sqrt_matmul_ceil.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sqrt_matmul_ceil: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sqrt_matmul_ceil: device mismatch");
}
Tensor fused_0414_sqrt_matmul_ceil(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x.matmul(y).sqrt()).ceil();
    if (!r.defined()) throw std::runtime_error("sqrt_matmul_ceil: undefined result");
    return r;
}
Tensor fused_0414_sqrt_matmul_ceil_contiguous(const Tensor& x,const Tensor& y) { return fused_0414_sqrt_matmul_ceil(x.contiguous(),y.contiguous()); }
Tensor fused_0414_sqrt_matmul_ceil_cpu(const Tensor& x,const Tensor& y) { return fused_0414_sqrt_matmul_ceil(x.cpu(),y.cpu()); }
Tensor fused_0414_sqrt_matmul_ceil_clone(const Tensor& x,const Tensor& y) { return fused_0414_sqrt_matmul_ceil(x.clone(),y.clone()); }
Tensor fused_0414_sqrt_matmul_ceil_detached(const Tensor& x,const Tensor& y) { return fused_0414_sqrt_matmul_ceil(x.detach(),y.detach()); }
bool fused_0414_sqrt_matmul_ceil_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0414_sqrt_matmul_ceil_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0414_sqrt_matmul_ceil_numel(const Tensor& x) { return x.numel(); }
DType fused_0414_sqrt_matmul_ceil_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0414_sqrt_matmul_ceil_device(const Tensor& x) { return x.device(); }
std::string fused_0414_sqrt_matmul_ceil_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0414_sqrt_matmul_ceil_name() { return "sqrt_matmul_ceil"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

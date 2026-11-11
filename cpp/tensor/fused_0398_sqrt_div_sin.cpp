#include "fused_0398_sqrt_div_sin.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sqrt_div_sin: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sqrt_div_sin: device mismatch");
}
Tensor fused_0398_sqrt_div_sin(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x / y.sqrt()).sin();
    if (!r.defined()) throw std::runtime_error("sqrt_div_sin: undefined result");
    return r;
}
Tensor fused_0398_sqrt_div_sin_contiguous(const Tensor& x,const Tensor& y) { return fused_0398_sqrt_div_sin(x.contiguous(),y.contiguous()); }
Tensor fused_0398_sqrt_div_sin_cpu(const Tensor& x,const Tensor& y) { return fused_0398_sqrt_div_sin(x.cpu(),y.cpu()); }
Tensor fused_0398_sqrt_div_sin_clone(const Tensor& x,const Tensor& y) { return fused_0398_sqrt_div_sin(x.clone(),y.clone()); }
Tensor fused_0398_sqrt_div_sin_detached(const Tensor& x,const Tensor& y) { return fused_0398_sqrt_div_sin(x.detach(),y.detach()); }
bool fused_0398_sqrt_div_sin_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0398_sqrt_div_sin_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0398_sqrt_div_sin_numel(const Tensor& x) { return x.numel(); }
DType fused_0398_sqrt_div_sin_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0398_sqrt_div_sin_device(const Tensor& x) { return x.device(); }
std::string fused_0398_sqrt_div_sin_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0398_sqrt_div_sin_name() { return "sqrt_div_sin"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

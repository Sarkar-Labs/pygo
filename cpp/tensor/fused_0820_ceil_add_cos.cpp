#include "fused_0820_ceil_add_cos.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("ceil_add_cos: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("ceil_add_cos: device mismatch");
}
Tensor fused_0820_ceil_add_cos(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x + y.ceil()).cos();
    if (!r.defined()) throw std::runtime_error("ceil_add_cos: undefined result");
    return r;
}
Tensor fused_0820_ceil_add_cos_contiguous(const Tensor& x,const Tensor& y) { return fused_0820_ceil_add_cos(x.contiguous(),y.contiguous()); }
Tensor fused_0820_ceil_add_cos_cpu(const Tensor& x,const Tensor& y) { return fused_0820_ceil_add_cos(x.cpu(),y.cpu()); }
Tensor fused_0820_ceil_add_cos_clone(const Tensor& x,const Tensor& y) { return fused_0820_ceil_add_cos(x.clone(),y.clone()); }
Tensor fused_0820_ceil_add_cos_detached(const Tensor& x,const Tensor& y) { return fused_0820_ceil_add_cos(x.detach(),y.detach()); }
bool fused_0820_ceil_add_cos_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0820_ceil_add_cos_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0820_ceil_add_cos_numel(const Tensor& x) { return x.numel(); }
DType fused_0820_ceil_add_cos_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0820_ceil_add_cos_device(const Tensor& x) { return x.device(); }
std::string fused_0820_ceil_add_cos_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0820_ceil_add_cos_name() { return "ceil_add_cos"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

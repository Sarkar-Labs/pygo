#include "fused_1006_gelu_sub_abs.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("gelu_sub_abs: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("gelu_sub_abs: device mismatch");
}
Tensor fused_1006_gelu_sub_abs(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x - y.gelu()).abs();
    if (!r.defined()) throw std::runtime_error("gelu_sub_abs: undefined result");
    return r;
}
Tensor fused_1006_gelu_sub_abs_contiguous(const Tensor& x,const Tensor& y) { return fused_1006_gelu_sub_abs(x.contiguous(),y.contiguous()); }
Tensor fused_1006_gelu_sub_abs_cpu(const Tensor& x,const Tensor& y) { return fused_1006_gelu_sub_abs(x.cpu(),y.cpu()); }
Tensor fused_1006_gelu_sub_abs_clone(const Tensor& x,const Tensor& y) { return fused_1006_gelu_sub_abs(x.clone(),y.clone()); }
Tensor fused_1006_gelu_sub_abs_detached(const Tensor& x,const Tensor& y) { return fused_1006_gelu_sub_abs(x.detach(),y.detach()); }
bool fused_1006_gelu_sub_abs_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1006_gelu_sub_abs_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_1006_gelu_sub_abs_numel(const Tensor& x) { return x.numel(); }
DType fused_1006_gelu_sub_abs_dtype(const Tensor& x) { return x.dtype(); }
Device fused_1006_gelu_sub_abs_device(const Tensor& x) { return x.device(); }
std::string fused_1006_gelu_sub_abs_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1006_gelu_sub_abs_name() { return "gelu_sub_abs"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

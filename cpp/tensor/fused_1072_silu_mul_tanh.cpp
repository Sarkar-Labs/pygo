#include "fused_1072_silu_mul_tanh.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("silu_mul_tanh: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("silu_mul_tanh: device mismatch");
}
Tensor fused_1072_silu_mul_tanh(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x * y.silu()).tanh();
    if (!r.defined()) throw std::runtime_error("silu_mul_tanh: undefined result");
    return r;
}
Tensor fused_1072_silu_mul_tanh_contiguous(const Tensor& x,const Tensor& y) { return fused_1072_silu_mul_tanh(x.contiguous(),y.contiguous()); }
Tensor fused_1072_silu_mul_tanh_cpu(const Tensor& x,const Tensor& y) { return fused_1072_silu_mul_tanh(x.cpu(),y.cpu()); }
Tensor fused_1072_silu_mul_tanh_clone(const Tensor& x,const Tensor& y) { return fused_1072_silu_mul_tanh(x.clone(),y.clone()); }
Tensor fused_1072_silu_mul_tanh_detached(const Tensor& x,const Tensor& y) { return fused_1072_silu_mul_tanh(x.detach(),y.detach()); }
bool fused_1072_silu_mul_tanh_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1072_silu_mul_tanh_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_1072_silu_mul_tanh_numel(const Tensor& x) { return x.numel(); }
DType fused_1072_silu_mul_tanh_dtype(const Tensor& x) { return x.dtype(); }
Device fused_1072_silu_mul_tanh_device(const Tensor& x) { return x.device(); }
std::string fused_1072_silu_mul_tanh_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1072_silu_mul_tanh_name() { return "silu_mul_tanh"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

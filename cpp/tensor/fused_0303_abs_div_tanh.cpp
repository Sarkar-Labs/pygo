#include "fused_0303_abs_div_tanh.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("abs_div_tanh: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("abs_div_tanh: device mismatch");
}
Tensor fused_0303_abs_div_tanh(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x / y.abs()).tanh();
    if (!r.defined()) throw std::runtime_error("abs_div_tanh: undefined result");
    return r;
}
Tensor fused_0303_abs_div_tanh_contiguous(const Tensor& x,const Tensor& y) { return fused_0303_abs_div_tanh(x.contiguous(),y.contiguous()); }
Tensor fused_0303_abs_div_tanh_cpu(const Tensor& x,const Tensor& y) { return fused_0303_abs_div_tanh(x.cpu(),y.cpu()); }
Tensor fused_0303_abs_div_tanh_clone(const Tensor& x,const Tensor& y) { return fused_0303_abs_div_tanh(x.clone(),y.clone()); }
Tensor fused_0303_abs_div_tanh_detached(const Tensor& x,const Tensor& y) { return fused_0303_abs_div_tanh(x.detach(),y.detach()); }
bool fused_0303_abs_div_tanh_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0303_abs_div_tanh_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0303_abs_div_tanh_numel(const Tensor& x) { return x.numel(); }
DType fused_0303_abs_div_tanh_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0303_abs_div_tanh_device(const Tensor& x) { return x.device(); }
std::string fused_0303_abs_div_tanh_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0303_abs_div_tanh_name() { return "abs_div_tanh"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

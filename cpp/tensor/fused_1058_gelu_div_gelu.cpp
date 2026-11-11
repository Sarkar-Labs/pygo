#include "fused_1058_gelu_div_gelu.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("gelu_div_gelu: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("gelu_div_gelu: device mismatch");
}
Tensor fused_1058_gelu_div_gelu(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x / y.gelu()).gelu();
    if (!r.defined()) throw std::runtime_error("gelu_div_gelu: undefined result");
    return r;
}
Tensor fused_1058_gelu_div_gelu_contiguous(const Tensor& x,const Tensor& y) { return fused_1058_gelu_div_gelu(x.contiguous(),y.contiguous()); }
Tensor fused_1058_gelu_div_gelu_cpu(const Tensor& x,const Tensor& y) { return fused_1058_gelu_div_gelu(x.cpu(),y.cpu()); }
Tensor fused_1058_gelu_div_gelu_clone(const Tensor& x,const Tensor& y) { return fused_1058_gelu_div_gelu(x.clone(),y.clone()); }
Tensor fused_1058_gelu_div_gelu_detached(const Tensor& x,const Tensor& y) { return fused_1058_gelu_div_gelu(x.detach(),y.detach()); }
bool fused_1058_gelu_div_gelu_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1058_gelu_div_gelu_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_1058_gelu_div_gelu_numel(const Tensor& x) { return x.numel(); }
DType fused_1058_gelu_div_gelu_dtype(const Tensor& x) { return x.dtype(); }
Device fused_1058_gelu_div_gelu_device(const Tensor& x) { return x.device(); }
std::string fused_1058_gelu_div_gelu_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1058_gelu_div_gelu_name() { return "gelu_div_gelu"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

#include "fused_0216_sigmoid_sub_gelu.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sigmoid_sub_gelu: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sigmoid_sub_gelu: device mismatch");
}
Tensor fused_0216_sigmoid_sub_gelu(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x - y.sigmoid()).gelu();
    if (!r.defined()) throw std::runtime_error("sigmoid_sub_gelu: undefined result");
    return r;
}
Tensor fused_0216_sigmoid_sub_gelu_contiguous(const Tensor& x,const Tensor& y) { return fused_0216_sigmoid_sub_gelu(x.contiguous(),y.contiguous()); }
Tensor fused_0216_sigmoid_sub_gelu_cpu(const Tensor& x,const Tensor& y) { return fused_0216_sigmoid_sub_gelu(x.cpu(),y.cpu()); }
Tensor fused_0216_sigmoid_sub_gelu_clone(const Tensor& x,const Tensor& y) { return fused_0216_sigmoid_sub_gelu(x.clone(),y.clone()); }
Tensor fused_0216_sigmoid_sub_gelu_detached(const Tensor& x,const Tensor& y) { return fused_0216_sigmoid_sub_gelu(x.detach(),y.detach()); }
bool fused_0216_sigmoid_sub_gelu_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0216_sigmoid_sub_gelu_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0216_sigmoid_sub_gelu_numel(const Tensor& x) { return x.numel(); }
DType fused_0216_sigmoid_sub_gelu_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0216_sigmoid_sub_gelu_device(const Tensor& x) { return x.device(); }
std::string fused_0216_sigmoid_sub_gelu_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0216_sigmoid_sub_gelu_name() { return "sigmoid_sub_gelu"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

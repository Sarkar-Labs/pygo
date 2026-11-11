#include "fused_0182_sigmoid_mul_log.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sigmoid_mul_log: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sigmoid_mul_log: device mismatch");
}
Tensor fused_0182_sigmoid_mul_log(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x * y.sigmoid()).log();
    if (!r.defined()) throw std::runtime_error("sigmoid_mul_log: undefined result");
    return r;
}
Tensor fused_0182_sigmoid_mul_log_contiguous(const Tensor& x,const Tensor& y) { return fused_0182_sigmoid_mul_log(x.contiguous(),y.contiguous()); }
Tensor fused_0182_sigmoid_mul_log_cpu(const Tensor& x,const Tensor& y) { return fused_0182_sigmoid_mul_log(x.cpu(),y.cpu()); }
Tensor fused_0182_sigmoid_mul_log_clone(const Tensor& x,const Tensor& y) { return fused_0182_sigmoid_mul_log(x.clone(),y.clone()); }
Tensor fused_0182_sigmoid_mul_log_detached(const Tensor& x,const Tensor& y) { return fused_0182_sigmoid_mul_log(x.detach(),y.detach()); }
bool fused_0182_sigmoid_mul_log_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0182_sigmoid_mul_log_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0182_sigmoid_mul_log_numel(const Tensor& x) { return x.numel(); }
DType fused_0182_sigmoid_mul_log_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0182_sigmoid_mul_log_device(const Tensor& x) { return x.device(); }
std::string fused_0182_sigmoid_mul_log_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0182_sigmoid_mul_log_name() { return "sigmoid_mul_log"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

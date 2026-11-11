#include "fused_0564_log_matmul_sign.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log_matmul_sign: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log_matmul_sign: device mismatch");
}
Tensor fused_0564_log_matmul_sign(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x.matmul(y).log()).sign();
    if (!r.defined()) throw std::runtime_error("log_matmul_sign: undefined result");
    return r;
}
Tensor fused_0564_log_matmul_sign_contiguous(const Tensor& x,const Tensor& y) { return fused_0564_log_matmul_sign(x.contiguous(),y.contiguous()); }
Tensor fused_0564_log_matmul_sign_cpu(const Tensor& x,const Tensor& y) { return fused_0564_log_matmul_sign(x.cpu(),y.cpu()); }
Tensor fused_0564_log_matmul_sign_clone(const Tensor& x,const Tensor& y) { return fused_0564_log_matmul_sign(x.clone(),y.clone()); }
Tensor fused_0564_log_matmul_sign_detached(const Tensor& x,const Tensor& y) { return fused_0564_log_matmul_sign(x.detach(),y.detach()); }
bool fused_0564_log_matmul_sign_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0564_log_matmul_sign_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0564_log_matmul_sign_numel(const Tensor& x) { return x.numel(); }
DType fused_0564_log_matmul_sign_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0564_log_matmul_sign_device(const Tensor& x) { return x.device(); }
std::string fused_0564_log_matmul_sign_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0564_log_matmul_sign_name() { return "log_matmul_sign"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

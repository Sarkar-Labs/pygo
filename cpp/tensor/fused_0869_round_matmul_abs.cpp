#include "fused_0869_round_matmul_abs.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("round_matmul_abs: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("round_matmul_abs: device mismatch");
}
Tensor fused_0869_round_matmul_abs(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x.matmul(y).round()).abs();
    if (!r.defined()) throw std::runtime_error("round_matmul_abs: undefined result");
    return r;
}
Tensor fused_0869_round_matmul_abs_contiguous(const Tensor& x,const Tensor& y) { return fused_0869_round_matmul_abs(x.contiguous(),y.contiguous()); }
Tensor fused_0869_round_matmul_abs_cpu(const Tensor& x,const Tensor& y) { return fused_0869_round_matmul_abs(x.cpu(),y.cpu()); }
Tensor fused_0869_round_matmul_abs_clone(const Tensor& x,const Tensor& y) { return fused_0869_round_matmul_abs(x.clone(),y.clone()); }
Tensor fused_0869_round_matmul_abs_detached(const Tensor& x,const Tensor& y) { return fused_0869_round_matmul_abs(x.detach(),y.detach()); }
bool fused_0869_round_matmul_abs_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0869_round_matmul_abs_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0869_round_matmul_abs_numel(const Tensor& x) { return x.numel(); }
DType fused_0869_round_matmul_abs_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0869_round_matmul_abs_device(const Tensor& x) { return x.device(); }
std::string fused_0869_round_matmul_abs_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0869_round_matmul_abs_name() { return "round_matmul_abs"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

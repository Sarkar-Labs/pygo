#include "fused_0179_sigmoid_matmul_exp.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sigmoid_matmul_exp: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sigmoid_matmul_exp: device mismatch");
}
Tensor fused_0179_sigmoid_matmul_exp(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x.matmul(y).sigmoid()).exp();
    if (!r.defined()) throw std::runtime_error("sigmoid_matmul_exp: undefined result");
    return r;
}
Tensor fused_0179_sigmoid_matmul_exp_contiguous(const Tensor& x,const Tensor& y) { return fused_0179_sigmoid_matmul_exp(x.contiguous(),y.contiguous()); }
Tensor fused_0179_sigmoid_matmul_exp_cpu(const Tensor& x,const Tensor& y) { return fused_0179_sigmoid_matmul_exp(x.cpu(),y.cpu()); }
Tensor fused_0179_sigmoid_matmul_exp_clone(const Tensor& x,const Tensor& y) { return fused_0179_sigmoid_matmul_exp(x.clone(),y.clone()); }
Tensor fused_0179_sigmoid_matmul_exp_detached(const Tensor& x,const Tensor& y) { return fused_0179_sigmoid_matmul_exp(x.detach(),y.detach()); }
bool fused_0179_sigmoid_matmul_exp_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0179_sigmoid_matmul_exp_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0179_sigmoid_matmul_exp_numel(const Tensor& x) { return x.numel(); }
DType fused_0179_sigmoid_matmul_exp_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0179_sigmoid_matmul_exp_device(const Tensor& x) { return x.device(); }
std::string fused_0179_sigmoid_matmul_exp_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0179_sigmoid_matmul_exp_name() { return "sigmoid_matmul_exp"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

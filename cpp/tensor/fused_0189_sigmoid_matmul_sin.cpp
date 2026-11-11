#include "fused_0189_sigmoid_matmul_sin.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sigmoid_matmul_sin: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sigmoid_matmul_sin: device mismatch");
}
Tensor fused_0189_sigmoid_matmul_sin(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x.matmul(y).sigmoid()).sin();
    if (!r.defined()) throw std::runtime_error("sigmoid_matmul_sin: undefined result");
    return r;
}
Tensor fused_0189_sigmoid_matmul_sin_contiguous(const Tensor& x,const Tensor& y) { return fused_0189_sigmoid_matmul_sin(x.contiguous(),y.contiguous()); }
Tensor fused_0189_sigmoid_matmul_sin_cpu(const Tensor& x,const Tensor& y) { return fused_0189_sigmoid_matmul_sin(x.cpu(),y.cpu()); }
Tensor fused_0189_sigmoid_matmul_sin_clone(const Tensor& x,const Tensor& y) { return fused_0189_sigmoid_matmul_sin(x.clone(),y.clone()); }
Tensor fused_0189_sigmoid_matmul_sin_detached(const Tensor& x,const Tensor& y) { return fused_0189_sigmoid_matmul_sin(x.detach(),y.detach()); }
bool fused_0189_sigmoid_matmul_sin_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0189_sigmoid_matmul_sin_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0189_sigmoid_matmul_sin_numel(const Tensor& x) { return x.numel(); }
DType fused_0189_sigmoid_matmul_sin_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0189_sigmoid_matmul_sin_device(const Tensor& x) { return x.device(); }
std::string fused_0189_sigmoid_matmul_sin_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0189_sigmoid_matmul_sin_name() { return "sigmoid_matmul_sin"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

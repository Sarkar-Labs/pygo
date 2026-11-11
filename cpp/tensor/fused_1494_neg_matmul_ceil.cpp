#include "fused_1494_neg_matmul_ceil.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1494_neg_matmul_ceil(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("neg_matmul_ceil: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("neg_matmul_ceil: device mismatch");
}
Tensor fused_1494_neg_matmul_ceil(const Tensor& x,const Tensor& y) { validate_fused_1494_neg_matmul_ceil(x,y); Tensor r=((x.matmul(y)).neg()).ceil(); if(!r.defined()) throw std::runtime_error("neg_matmul_ceil: undefined result"); return r; }
Tensor fused_1494_neg_matmul_ceil_contiguous(const Tensor& x,const Tensor& y) { return fused_1494_neg_matmul_ceil(x.contiguous(),y.contiguous()); }
Tensor fused_1494_neg_matmul_ceil_cpu(const Tensor& x,const Tensor& y) { return fused_1494_neg_matmul_ceil(x.cpu(),y.cpu()); }
Tensor fused_1494_neg_matmul_ceil_clone(const Tensor& x,const Tensor& y) { return fused_1494_neg_matmul_ceil(x.clone(),y.clone()); }
Tensor fused_1494_neg_matmul_ceil_detached(const Tensor& x,const Tensor& y) { return fused_1494_neg_matmul_ceil(x.detach(),y.detach()); }
bool fused_1494_neg_matmul_ceil_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1494_neg_matmul_ceil_numel(const Tensor& x) { return x.numel(); }
std::string fused_1494_neg_matmul_ceil_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1494_neg_matmul_ceil_name() { return "neg_matmul_ceil"; }
}

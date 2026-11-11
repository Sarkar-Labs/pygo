#include "fused_1359_silu_matmul_sigmoid.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1359_silu_matmul_sigmoid(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("silu_matmul_sigmoid: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("silu_matmul_sigmoid: device mismatch");
}
Tensor fused_1359_silu_matmul_sigmoid(const Tensor& x,const Tensor& y) { validate_fused_1359_silu_matmul_sigmoid(x,y); Tensor r=((x.matmul(y)).silu()).sigmoid(); if(!r.defined()) throw std::runtime_error("silu_matmul_sigmoid: undefined result"); return r; }
Tensor fused_1359_silu_matmul_sigmoid_contiguous(const Tensor& x,const Tensor& y) { return fused_1359_silu_matmul_sigmoid(x.contiguous(),y.contiguous()); }
Tensor fused_1359_silu_matmul_sigmoid_cpu(const Tensor& x,const Tensor& y) { return fused_1359_silu_matmul_sigmoid(x.cpu(),y.cpu()); }
Tensor fused_1359_silu_matmul_sigmoid_clone(const Tensor& x,const Tensor& y) { return fused_1359_silu_matmul_sigmoid(x.clone(),y.clone()); }
Tensor fused_1359_silu_matmul_sigmoid_detached(const Tensor& x,const Tensor& y) { return fused_1359_silu_matmul_sigmoid(x.detach(),y.detach()); }
bool fused_1359_silu_matmul_sigmoid_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1359_silu_matmul_sigmoid_numel(const Tensor& x) { return x.numel(); }
std::string fused_1359_silu_matmul_sigmoid_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1359_silu_matmul_sigmoid_name() { return "silu_matmul_sigmoid"; }
}

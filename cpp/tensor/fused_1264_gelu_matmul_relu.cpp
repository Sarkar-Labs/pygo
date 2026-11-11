#include "fused_1264_gelu_matmul_relu.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1264_gelu_matmul_relu(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("gelu_matmul_relu: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("gelu_matmul_relu: device mismatch");
}
Tensor fused_1264_gelu_matmul_relu(const Tensor& x,const Tensor& y) { validate_fused_1264_gelu_matmul_relu(x,y); Tensor r=((x.matmul(y)).gelu()).relu(); if(!r.defined()) throw std::runtime_error("gelu_matmul_relu: undefined result"); return r; }
Tensor fused_1264_gelu_matmul_relu_contiguous(const Tensor& x,const Tensor& y) { return fused_1264_gelu_matmul_relu(x.contiguous(),y.contiguous()); }
Tensor fused_1264_gelu_matmul_relu_cpu(const Tensor& x,const Tensor& y) { return fused_1264_gelu_matmul_relu(x.cpu(),y.cpu()); }
Tensor fused_1264_gelu_matmul_relu_clone(const Tensor& x,const Tensor& y) { return fused_1264_gelu_matmul_relu(x.clone(),y.clone()); }
Tensor fused_1264_gelu_matmul_relu_detached(const Tensor& x,const Tensor& y) { return fused_1264_gelu_matmul_relu(x.detach(),y.detach()); }
bool fused_1264_gelu_matmul_relu_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1264_gelu_matmul_relu_numel(const Tensor& x) { return x.numel(); }
std::string fused_1264_gelu_matmul_relu_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1264_gelu_matmul_relu_name() { return "gelu_matmul_relu"; }
}

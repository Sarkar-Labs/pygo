#include "fused_0319_tanh_matmul_floor.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0319_tanh_matmul_floor(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("tanh_matmul_floor: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("tanh_matmul_floor: device mismatch");
}
Tensor fused_0319_tanh_matmul_floor(const Tensor& x,const Tensor& y) { validate_fused_0319_tanh_matmul_floor(x,y); Tensor r=((x.matmul(y)).tanh()).floor(); if(!r.defined()) throw std::runtime_error("tanh_matmul_floor: undefined result"); return r; }
Tensor fused_0319_tanh_matmul_floor_contiguous(const Tensor& x,const Tensor& y) { return fused_0319_tanh_matmul_floor(x.contiguous(),y.contiguous()); }
Tensor fused_0319_tanh_matmul_floor_cpu(const Tensor& x,const Tensor& y) { return fused_0319_tanh_matmul_floor(x.cpu(),y.cpu()); }
Tensor fused_0319_tanh_matmul_floor_clone(const Tensor& x,const Tensor& y) { return fused_0319_tanh_matmul_floor(x.clone(),y.clone()); }
Tensor fused_0319_tanh_matmul_floor_detached(const Tensor& x,const Tensor& y) { return fused_0319_tanh_matmul_floor(x.detach(),y.detach()); }
bool fused_0319_tanh_matmul_floor_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0319_tanh_matmul_floor_numel(const Tensor& x) { return x.numel(); }
std::string fused_0319_tanh_matmul_floor_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0319_tanh_matmul_floor_name() { return "tanh_matmul_floor"; }
}

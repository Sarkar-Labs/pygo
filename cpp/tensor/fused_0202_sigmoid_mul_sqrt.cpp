#include "fused_0202_sigmoid_mul_sqrt.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_0202_sigmoid_mul_sqrt(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sigmoid_mul_sqrt: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sigmoid_mul_sqrt: device mismatch");
}
Tensor fused_0202_sigmoid_mul_sqrt(const Tensor& x,const Tensor& y) { validate_fused_0202_sigmoid_mul_sqrt(x,y); Tensor r=((x * y).sigmoid()).sqrt(); if(!r.defined()) throw std::runtime_error("sigmoid_mul_sqrt: undefined result"); return r; }
Tensor fused_0202_sigmoid_mul_sqrt_contiguous(const Tensor& x,const Tensor& y) { return fused_0202_sigmoid_mul_sqrt(x.contiguous(),y.contiguous()); }
Tensor fused_0202_sigmoid_mul_sqrt_cpu(const Tensor& x,const Tensor& y) { return fused_0202_sigmoid_mul_sqrt(x.cpu(),y.cpu()); }
Tensor fused_0202_sigmoid_mul_sqrt_clone(const Tensor& x,const Tensor& y) { return fused_0202_sigmoid_mul_sqrt(x.clone(),y.clone()); }
Tensor fused_0202_sigmoid_mul_sqrt_detached(const Tensor& x,const Tensor& y) { return fused_0202_sigmoid_mul_sqrt(x.detach(),y.detach()); }
bool fused_0202_sigmoid_mul_sqrt_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0202_sigmoid_mul_sqrt_numel(const Tensor& x) { return x.numel(); }
std::string fused_0202_sigmoid_mul_sqrt_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0202_sigmoid_mul_sqrt_name() { return "sigmoid_mul_sqrt"; }
}

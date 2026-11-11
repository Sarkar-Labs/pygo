#include "fused_1010_ceil_add_sqrt.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1010_ceil_add_sqrt(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("ceil_add_sqrt: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("ceil_add_sqrt: device mismatch");
}
Tensor fused_1010_ceil_add_sqrt(const Tensor& x,const Tensor& y) { validate_fused_1010_ceil_add_sqrt(x,y); Tensor r=((x + y).ceil()).sqrt(); if(!r.defined()) throw std::runtime_error("ceil_add_sqrt: undefined result"); return r; }
Tensor fused_1010_ceil_add_sqrt_contiguous(const Tensor& x,const Tensor& y) { return fused_1010_ceil_add_sqrt(x.contiguous(),y.contiguous()); }
Tensor fused_1010_ceil_add_sqrt_cpu(const Tensor& x,const Tensor& y) { return fused_1010_ceil_add_sqrt(x.cpu(),y.cpu()); }
Tensor fused_1010_ceil_add_sqrt_clone(const Tensor& x,const Tensor& y) { return fused_1010_ceil_add_sqrt(x.clone(),y.clone()); }
Tensor fused_1010_ceil_add_sqrt_detached(const Tensor& x,const Tensor& y) { return fused_1010_ceil_add_sqrt(x.detach(),y.detach()); }
bool fused_1010_ceil_add_sqrt_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1010_ceil_add_sqrt_numel(const Tensor& x) { return x.numel(); }
std::string fused_1010_ceil_add_sqrt_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1010_ceil_add_sqrt_name() { return "ceil_add_sqrt"; }
}

#include "fused_1215_sign_add_floor.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1215_sign_add_floor(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sign_add_floor: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sign_add_floor: device mismatch");
}
Tensor fused_1215_sign_add_floor(const Tensor& x,const Tensor& y) { validate_fused_1215_sign_add_floor(x,y); Tensor r=((x + y).sign()).floor(); if(!r.defined()) throw std::runtime_error("sign_add_floor: undefined result"); return r; }
Tensor fused_1215_sign_add_floor_contiguous(const Tensor& x,const Tensor& y) { return fused_1215_sign_add_floor(x.contiguous(),y.contiguous()); }
Tensor fused_1215_sign_add_floor_cpu(const Tensor& x,const Tensor& y) { return fused_1215_sign_add_floor(x.cpu(),y.cpu()); }
Tensor fused_1215_sign_add_floor_clone(const Tensor& x,const Tensor& y) { return fused_1215_sign_add_floor(x.clone(),y.clone()); }
Tensor fused_1215_sign_add_floor_detached(const Tensor& x,const Tensor& y) { return fused_1215_sign_add_floor(x.detach(),y.detach()); }
bool fused_1215_sign_add_floor_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1215_sign_add_floor_numel(const Tensor& x) { return x.numel(); }
std::string fused_1215_sign_add_floor_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1215_sign_add_floor_name() { return "sign_add_floor"; }
}

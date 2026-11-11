#include "fused_1445_neg_add_sigmoid.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1445_neg_add_sigmoid(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("neg_add_sigmoid: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("neg_add_sigmoid: device mismatch");
}
Tensor fused_1445_neg_add_sigmoid(const Tensor& x,const Tensor& y) { validate_fused_1445_neg_add_sigmoid(x,y); Tensor r=((x + y).neg()).sigmoid(); if(!r.defined()) throw std::runtime_error("neg_add_sigmoid: undefined result"); return r; }
Tensor fused_1445_neg_add_sigmoid_contiguous(const Tensor& x,const Tensor& y) { return fused_1445_neg_add_sigmoid(x.contiguous(),y.contiguous()); }
Tensor fused_1445_neg_add_sigmoid_cpu(const Tensor& x,const Tensor& y) { return fused_1445_neg_add_sigmoid(x.cpu(),y.cpu()); }
Tensor fused_1445_neg_add_sigmoid_clone(const Tensor& x,const Tensor& y) { return fused_1445_neg_add_sigmoid(x.clone(),y.clone()); }
Tensor fused_1445_neg_add_sigmoid_detached(const Tensor& x,const Tensor& y) { return fused_1445_neg_add_sigmoid(x.detach(),y.detach()); }
bool fused_1445_neg_add_sigmoid_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1445_neg_add_sigmoid_numel(const Tensor& x) { return x.numel(); }
std::string fused_1445_neg_add_sigmoid_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1445_neg_add_sigmoid_name() { return "neg_add_sigmoid"; }
}

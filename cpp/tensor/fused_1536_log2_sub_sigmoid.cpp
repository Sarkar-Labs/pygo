#include "fused_1536_log2_sub_sigmoid.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1536_log2_sub_sigmoid(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log2_sub_sigmoid: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log2_sub_sigmoid: device mismatch");
}
Tensor fused_1536_log2_sub_sigmoid(const Tensor& x,const Tensor& y) { validate_fused_1536_log2_sub_sigmoid(x,y); Tensor r=((x - y).log2()).sigmoid(); if(!r.defined()) throw std::runtime_error("log2_sub_sigmoid: undefined result"); return r; }
Tensor fused_1536_log2_sub_sigmoid_contiguous(const Tensor& x,const Tensor& y) { return fused_1536_log2_sub_sigmoid(x.contiguous(),y.contiguous()); }
Tensor fused_1536_log2_sub_sigmoid_cpu(const Tensor& x,const Tensor& y) { return fused_1536_log2_sub_sigmoid(x.cpu(),y.cpu()); }
Tensor fused_1536_log2_sub_sigmoid_clone(const Tensor& x,const Tensor& y) { return fused_1536_log2_sub_sigmoid(x.clone(),y.clone()); }
Tensor fused_1536_log2_sub_sigmoid_detached(const Tensor& x,const Tensor& y) { return fused_1536_log2_sub_sigmoid(x.detach(),y.detach()); }
bool fused_1536_log2_sub_sigmoid_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1536_log2_sub_sigmoid_numel(const Tensor& x) { return x.numel(); }
std::string fused_1536_log2_sub_sigmoid_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1536_log2_sub_sigmoid_name() { return "log2_sub_sigmoid"; }
}

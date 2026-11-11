#include "fused_1710_sum_flatten.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1710_sum_flatten(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sum_flatten: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sum_flatten: device mismatch");
}
Tensor fused_1710_sum_flatten(const Tensor& x,const Tensor& y) { validate_fused_1710_sum_flatten(x,y); Tensor r=x.sum(0); if(!r.defined()) throw std::runtime_error("sum_flatten: undefined result"); return r; }
Tensor fused_1710_sum_flatten_contiguous(const Tensor& x,const Tensor& y) { return fused_1710_sum_flatten(x.contiguous(),y.contiguous()); }
Tensor fused_1710_sum_flatten_cpu(const Tensor& x,const Tensor& y) { return fused_1710_sum_flatten(x.cpu(),y.cpu()); }
Tensor fused_1710_sum_flatten_clone(const Tensor& x,const Tensor& y) { return fused_1710_sum_flatten(x.clone(),y.clone()); }
Tensor fused_1710_sum_flatten_detached(const Tensor& x,const Tensor& y) { return fused_1710_sum_flatten(x.detach(),y.detach()); }
bool fused_1710_sum_flatten_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1710_sum_flatten_numel(const Tensor& x) { return x.numel(); }
std::string fused_1710_sum_flatten_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1710_sum_flatten_name() { return "sum_flatten"; }
}

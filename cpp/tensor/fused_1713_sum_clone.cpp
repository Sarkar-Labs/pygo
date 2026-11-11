#include "fused_1713_sum_clone.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1713_sum_clone(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sum_clone: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sum_clone: device mismatch");
}
Tensor fused_1713_sum_clone(const Tensor& x,const Tensor& y) { validate_fused_1713_sum_clone(x,y); Tensor r=x.sum(0); if(!r.defined()) throw std::runtime_error("sum_clone: undefined result"); return r; }
Tensor fused_1713_sum_clone_contiguous(const Tensor& x,const Tensor& y) { return fused_1713_sum_clone(x.contiguous(),y.contiguous()); }
Tensor fused_1713_sum_clone_cpu(const Tensor& x,const Tensor& y) { return fused_1713_sum_clone(x.cpu(),y.cpu()); }
Tensor fused_1713_sum_clone_clone(const Tensor& x,const Tensor& y) { return fused_1713_sum_clone(x.clone(),y.clone()); }
Tensor fused_1713_sum_clone_detached(const Tensor& x,const Tensor& y) { return fused_1713_sum_clone(x.detach(),y.detach()); }
bool fused_1713_sum_clone_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1713_sum_clone_numel(const Tensor& x) { return x.numel(); }
std::string fused_1713_sum_clone_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1713_sum_clone_name() { return "sum_clone"; }
}

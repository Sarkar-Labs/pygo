#include "fused_1712_sum_unsqueeze.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1712_sum_unsqueeze(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sum_unsqueeze: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sum_unsqueeze: device mismatch");
}
Tensor fused_1712_sum_unsqueeze(const Tensor& x,const Tensor& y) { validate_fused_1712_sum_unsqueeze(x,y); Tensor r=x.sum(0); if(!r.defined()) throw std::runtime_error("sum_unsqueeze: undefined result"); return r; }
Tensor fused_1712_sum_unsqueeze_contiguous(const Tensor& x,const Tensor& y) { return fused_1712_sum_unsqueeze(x.contiguous(),y.contiguous()); }
Tensor fused_1712_sum_unsqueeze_cpu(const Tensor& x,const Tensor& y) { return fused_1712_sum_unsqueeze(x.cpu(),y.cpu()); }
Tensor fused_1712_sum_unsqueeze_clone(const Tensor& x,const Tensor& y) { return fused_1712_sum_unsqueeze(x.clone(),y.clone()); }
Tensor fused_1712_sum_unsqueeze_detached(const Tensor& x,const Tensor& y) { return fused_1712_sum_unsqueeze(x.detach(),y.detach()); }
bool fused_1712_sum_unsqueeze_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1712_sum_unsqueeze_numel(const Tensor& x) { return x.numel(); }
std::string fused_1712_sum_unsqueeze_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1712_sum_unsqueeze_name() { return "sum_unsqueeze"; }
}

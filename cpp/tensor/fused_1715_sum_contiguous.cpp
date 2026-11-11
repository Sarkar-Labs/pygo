#include "fused_1715_sum_contiguous.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1715_sum_contiguous(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sum_contiguous: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sum_contiguous: device mismatch");
}
Tensor fused_1715_sum_contiguous(const Tensor& x,const Tensor& y) { validate_fused_1715_sum_contiguous(x,y); Tensor r=x.sum(0); if(!r.defined()) throw std::runtime_error("sum_contiguous: undefined result"); return r; }
Tensor fused_1715_sum_contiguous_contiguous(const Tensor& x,const Tensor& y) { return fused_1715_sum_contiguous(x.contiguous(),y.contiguous()); }
Tensor fused_1715_sum_contiguous_cpu(const Tensor& x,const Tensor& y) { return fused_1715_sum_contiguous(x.cpu(),y.cpu()); }
Tensor fused_1715_sum_contiguous_clone(const Tensor& x,const Tensor& y) { return fused_1715_sum_contiguous(x.clone(),y.clone()); }
Tensor fused_1715_sum_contiguous_detached(const Tensor& x,const Tensor& y) { return fused_1715_sum_contiguous(x.detach(),y.detach()); }
bool fused_1715_sum_contiguous_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1715_sum_contiguous_numel(const Tensor& x) { return x.numel(); }
std::string fused_1715_sum_contiguous_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1715_sum_contiguous_name() { return "sum_contiguous"; }
}

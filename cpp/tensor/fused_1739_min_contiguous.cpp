#include "fused_1739_min_contiguous.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1739_min_contiguous(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("min_contiguous: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("min_contiguous: device mismatch");
}
Tensor fused_1739_min_contiguous(const Tensor& x,const Tensor& y) { validate_fused_1739_min_contiguous(x,y); Tensor r=x.min(0); if(!r.defined()) throw std::runtime_error("min_contiguous: undefined result"); return r; }
Tensor fused_1739_min_contiguous_contiguous(const Tensor& x,const Tensor& y) { return fused_1739_min_contiguous(x.contiguous(),y.contiguous()); }
Tensor fused_1739_min_contiguous_cpu(const Tensor& x,const Tensor& y) { return fused_1739_min_contiguous(x.cpu(),y.cpu()); }
Tensor fused_1739_min_contiguous_clone(const Tensor& x,const Tensor& y) { return fused_1739_min_contiguous(x.clone(),y.clone()); }
Tensor fused_1739_min_contiguous_detached(const Tensor& x,const Tensor& y) { return fused_1739_min_contiguous(x.detach(),y.detach()); }
bool fused_1739_min_contiguous_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1739_min_contiguous_numel(const Tensor& x) { return x.numel(); }
std::string fused_1739_min_contiguous_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1739_min_contiguous_name() { return "min_contiguous"; }
}

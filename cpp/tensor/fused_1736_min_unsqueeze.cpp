#include "fused_1736_min_unsqueeze.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1736_min_unsqueeze(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("min_unsqueeze: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("min_unsqueeze: device mismatch");
}
Tensor fused_1736_min_unsqueeze(const Tensor& x,const Tensor& y) { validate_fused_1736_min_unsqueeze(x,y); Tensor r=x.min(0); if(!r.defined()) throw std::runtime_error("min_unsqueeze: undefined result"); return r; }
Tensor fused_1736_min_unsqueeze_contiguous(const Tensor& x,const Tensor& y) { return fused_1736_min_unsqueeze(x.contiguous(),y.contiguous()); }
Tensor fused_1736_min_unsqueeze_cpu(const Tensor& x,const Tensor& y) { return fused_1736_min_unsqueeze(x.cpu(),y.cpu()); }
Tensor fused_1736_min_unsqueeze_clone(const Tensor& x,const Tensor& y) { return fused_1736_min_unsqueeze(x.clone(),y.clone()); }
Tensor fused_1736_min_unsqueeze_detached(const Tensor& x,const Tensor& y) { return fused_1736_min_unsqueeze(x.detach(),y.detach()); }
bool fused_1736_min_unsqueeze_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1736_min_unsqueeze_numel(const Tensor& x) { return x.numel(); }
std::string fused_1736_min_unsqueeze_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1736_min_unsqueeze_name() { return "min_unsqueeze"; }
}

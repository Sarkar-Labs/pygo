#include "fused_1724_max_unsqueeze.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1724_max_unsqueeze(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("max_unsqueeze: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("max_unsqueeze: device mismatch");
}
Tensor fused_1724_max_unsqueeze(const Tensor& x,const Tensor& y) { validate_fused_1724_max_unsqueeze(x,y); Tensor r=x.max(0); if(!r.defined()) throw std::runtime_error("max_unsqueeze: undefined result"); return r; }
Tensor fused_1724_max_unsqueeze_contiguous(const Tensor& x,const Tensor& y) { return fused_1724_max_unsqueeze(x.contiguous(),y.contiguous()); }
Tensor fused_1724_max_unsqueeze_cpu(const Tensor& x,const Tensor& y) { return fused_1724_max_unsqueeze(x.cpu(),y.cpu()); }
Tensor fused_1724_max_unsqueeze_clone(const Tensor& x,const Tensor& y) { return fused_1724_max_unsqueeze(x.clone(),y.clone()); }
Tensor fused_1724_max_unsqueeze_detached(const Tensor& x,const Tensor& y) { return fused_1724_max_unsqueeze(x.detach(),y.detach()); }
bool fused_1724_max_unsqueeze_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1724_max_unsqueeze_numel(const Tensor& x) { return x.numel(); }
std::string fused_1724_max_unsqueeze_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1724_max_unsqueeze_name() { return "max_unsqueeze"; }
}

#include "fused_1730_argmax_unsqueeze.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1730_argmax_unsqueeze(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("argmax_unsqueeze: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("argmax_unsqueeze: device mismatch");
}
Tensor fused_1730_argmax_unsqueeze(const Tensor& x,const Tensor& y) { validate_fused_1730_argmax_unsqueeze(x,y); Tensor r=x.argmax(0); if(!r.defined()) throw std::runtime_error("argmax_unsqueeze: undefined result"); return r; }
Tensor fused_1730_argmax_unsqueeze_contiguous(const Tensor& x,const Tensor& y) { return fused_1730_argmax_unsqueeze(x.contiguous(),y.contiguous()); }
Tensor fused_1730_argmax_unsqueeze_cpu(const Tensor& x,const Tensor& y) { return fused_1730_argmax_unsqueeze(x.cpu(),y.cpu()); }
Tensor fused_1730_argmax_unsqueeze_clone(const Tensor& x,const Tensor& y) { return fused_1730_argmax_unsqueeze(x.clone(),y.clone()); }
Tensor fused_1730_argmax_unsqueeze_detached(const Tensor& x,const Tensor& y) { return fused_1730_argmax_unsqueeze(x.detach(),y.detach()); }
bool fused_1730_argmax_unsqueeze_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1730_argmax_unsqueeze_numel(const Tensor& x) { return x.numel(); }
std::string fused_1730_argmax_unsqueeze_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1730_argmax_unsqueeze_name() { return "argmax_unsqueeze"; }
}

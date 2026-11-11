#include "fused_1726_max_detach.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1726_max_detach(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("max_detach: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("max_detach: device mismatch");
}
Tensor fused_1726_max_detach(const Tensor& x,const Tensor& y) { validate_fused_1726_max_detach(x,y); Tensor r=x.max(0); if(!r.defined()) throw std::runtime_error("max_detach: undefined result"); return r; }
Tensor fused_1726_max_detach_contiguous(const Tensor& x,const Tensor& y) { return fused_1726_max_detach(x.contiguous(),y.contiguous()); }
Tensor fused_1726_max_detach_cpu(const Tensor& x,const Tensor& y) { return fused_1726_max_detach(x.cpu(),y.cpu()); }
Tensor fused_1726_max_detach_clone(const Tensor& x,const Tensor& y) { return fused_1726_max_detach(x.clone(),y.clone()); }
Tensor fused_1726_max_detach_detached(const Tensor& x,const Tensor& y) { return fused_1726_max_detach(x.detach(),y.detach()); }
bool fused_1726_max_detach_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1726_max_detach_numel(const Tensor& x) { return x.numel(); }
std::string fused_1726_max_detach_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1726_max_detach_name() { return "max_detach"; }
}

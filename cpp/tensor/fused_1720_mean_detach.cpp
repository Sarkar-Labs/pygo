#include "fused_1720_mean_detach.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1720_mean_detach(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("mean_detach: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("mean_detach: device mismatch");
}
Tensor fused_1720_mean_detach(const Tensor& x,const Tensor& y) { validate_fused_1720_mean_detach(x,y); Tensor r=x.mean(0); if(!r.defined()) throw std::runtime_error("mean_detach: undefined result"); return r; }
Tensor fused_1720_mean_detach_contiguous(const Tensor& x,const Tensor& y) { return fused_1720_mean_detach(x.contiguous(),y.contiguous()); }
Tensor fused_1720_mean_detach_cpu(const Tensor& x,const Tensor& y) { return fused_1720_mean_detach(x.cpu(),y.cpu()); }
Tensor fused_1720_mean_detach_clone(const Tensor& x,const Tensor& y) { return fused_1720_mean_detach(x.clone(),y.clone()); }
Tensor fused_1720_mean_detach_detached(const Tensor& x,const Tensor& y) { return fused_1720_mean_detach(x.detach(),y.detach()); }
bool fused_1720_mean_detach_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1720_mean_detach_numel(const Tensor& x) { return x.numel(); }
std::string fused_1720_mean_detach_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1720_mean_detach_name() { return "mean_detach"; }
}

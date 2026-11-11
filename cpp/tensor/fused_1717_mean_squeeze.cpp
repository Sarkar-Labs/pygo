#include "fused_1717_mean_squeeze.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1717_mean_squeeze(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("mean_squeeze: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("mean_squeeze: device mismatch");
}
Tensor fused_1717_mean_squeeze(const Tensor& x,const Tensor& y) { validate_fused_1717_mean_squeeze(x,y); Tensor r=x.mean(0); if(!r.defined()) throw std::runtime_error("mean_squeeze: undefined result"); return r; }
Tensor fused_1717_mean_squeeze_contiguous(const Tensor& x,const Tensor& y) { return fused_1717_mean_squeeze(x.contiguous(),y.contiguous()); }
Tensor fused_1717_mean_squeeze_cpu(const Tensor& x,const Tensor& y) { return fused_1717_mean_squeeze(x.cpu(),y.cpu()); }
Tensor fused_1717_mean_squeeze_clone(const Tensor& x,const Tensor& y) { return fused_1717_mean_squeeze(x.clone(),y.clone()); }
Tensor fused_1717_mean_squeeze_detached(const Tensor& x,const Tensor& y) { return fused_1717_mean_squeeze(x.detach(),y.detach()); }
bool fused_1717_mean_squeeze_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1717_mean_squeeze_numel(const Tensor& x) { return x.numel(); }
std::string fused_1717_mean_squeeze_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1717_mean_squeeze_name() { return "mean_squeeze"; }
}

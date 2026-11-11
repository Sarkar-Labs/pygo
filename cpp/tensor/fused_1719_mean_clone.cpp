#include "fused_1719_mean_clone.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1719_mean_clone(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("mean_clone: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("mean_clone: device mismatch");
}
Tensor fused_1719_mean_clone(const Tensor& x,const Tensor& y) { validate_fused_1719_mean_clone(x,y); Tensor r=x.mean(0); if(!r.defined()) throw std::runtime_error("mean_clone: undefined result"); return r; }
Tensor fused_1719_mean_clone_contiguous(const Tensor& x,const Tensor& y) { return fused_1719_mean_clone(x.contiguous(),y.contiguous()); }
Tensor fused_1719_mean_clone_cpu(const Tensor& x,const Tensor& y) { return fused_1719_mean_clone(x.cpu(),y.cpu()); }
Tensor fused_1719_mean_clone_clone(const Tensor& x,const Tensor& y) { return fused_1719_mean_clone(x.clone(),y.clone()); }
Tensor fused_1719_mean_clone_detached(const Tensor& x,const Tensor& y) { return fused_1719_mean_clone(x.detach(),y.detach()); }
bool fused_1719_mean_clone_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1719_mean_clone_numel(const Tensor& x) { return x.numel(); }
std::string fused_1719_mean_clone_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1719_mean_clone_name() { return "mean_clone"; }
}

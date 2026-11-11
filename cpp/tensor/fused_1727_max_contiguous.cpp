#include "fused_1727_max_contiguous.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1727_max_contiguous(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("max_contiguous: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("max_contiguous: device mismatch");
}
Tensor fused_1727_max_contiguous(const Tensor& x,const Tensor& y) { validate_fused_1727_max_contiguous(x,y); Tensor r=x.max(0); if(!r.defined()) throw std::runtime_error("max_contiguous: undefined result"); return r; }
Tensor fused_1727_max_contiguous_contiguous(const Tensor& x,const Tensor& y) { return fused_1727_max_contiguous(x.contiguous(),y.contiguous()); }
Tensor fused_1727_max_contiguous_cpu(const Tensor& x,const Tensor& y) { return fused_1727_max_contiguous(x.cpu(),y.cpu()); }
Tensor fused_1727_max_contiguous_clone(const Tensor& x,const Tensor& y) { return fused_1727_max_contiguous(x.clone(),y.clone()); }
Tensor fused_1727_max_contiguous_detached(const Tensor& x,const Tensor& y) { return fused_1727_max_contiguous(x.detach(),y.detach()); }
bool fused_1727_max_contiguous_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1727_max_contiguous_numel(const Tensor& x) { return x.numel(); }
std::string fused_1727_max_contiguous_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1727_max_contiguous_name() { return "max_contiguous"; }
}

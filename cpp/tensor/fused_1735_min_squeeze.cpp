#include "fused_1735_min_squeeze.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1735_min_squeeze(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("min_squeeze: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("min_squeeze: device mismatch");
}
Tensor fused_1735_min_squeeze(const Tensor& x,const Tensor& y) { validate_fused_1735_min_squeeze(x,y); Tensor r=x.min(0); if(!r.defined()) throw std::runtime_error("min_squeeze: undefined result"); return r; }
Tensor fused_1735_min_squeeze_contiguous(const Tensor& x,const Tensor& y) { return fused_1735_min_squeeze(x.contiguous(),y.contiguous()); }
Tensor fused_1735_min_squeeze_cpu(const Tensor& x,const Tensor& y) { return fused_1735_min_squeeze(x.cpu(),y.cpu()); }
Tensor fused_1735_min_squeeze_clone(const Tensor& x,const Tensor& y) { return fused_1735_min_squeeze(x.clone(),y.clone()); }
Tensor fused_1735_min_squeeze_detached(const Tensor& x,const Tensor& y) { return fused_1735_min_squeeze(x.detach(),y.detach()); }
bool fused_1735_min_squeeze_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1735_min_squeeze_numel(const Tensor& x) { return x.numel(); }
std::string fused_1735_min_squeeze_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1735_min_squeeze_name() { return "min_squeeze"; }
}

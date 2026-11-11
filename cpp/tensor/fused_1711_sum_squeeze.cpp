#include "fused_1711_sum_squeeze.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1711_sum_squeeze(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("sum_squeeze: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("sum_squeeze: device mismatch");
}
Tensor fused_1711_sum_squeeze(const Tensor& x,const Tensor& y) { validate_fused_1711_sum_squeeze(x,y); Tensor r=x.sum(0); if(!r.defined()) throw std::runtime_error("sum_squeeze: undefined result"); return r; }
Tensor fused_1711_sum_squeeze_contiguous(const Tensor& x,const Tensor& y) { return fused_1711_sum_squeeze(x.contiguous(),y.contiguous()); }
Tensor fused_1711_sum_squeeze_cpu(const Tensor& x,const Tensor& y) { return fused_1711_sum_squeeze(x.cpu(),y.cpu()); }
Tensor fused_1711_sum_squeeze_clone(const Tensor& x,const Tensor& y) { return fused_1711_sum_squeeze(x.clone(),y.clone()); }
Tensor fused_1711_sum_squeeze_detached(const Tensor& x,const Tensor& y) { return fused_1711_sum_squeeze(x.detach(),y.detach()); }
bool fused_1711_sum_squeeze_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1711_sum_squeeze_numel(const Tensor& x) { return x.numel(); }
std::string fused_1711_sum_squeeze_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1711_sum_squeeze_name() { return "sum_squeeze"; }
}

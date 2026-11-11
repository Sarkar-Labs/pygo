#include "fused_1729_argmax_squeeze.h"
#include <stdexcept>
namespace pygo::tensor_ops {
static void validate_fused_1729_argmax_squeeze(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("argmax_squeeze: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("argmax_squeeze: device mismatch");
}
Tensor fused_1729_argmax_squeeze(const Tensor& x,const Tensor& y) { validate_fused_1729_argmax_squeeze(x,y); Tensor r=x.argmax(0); if(!r.defined()) throw std::runtime_error("argmax_squeeze: undefined result"); return r; }
Tensor fused_1729_argmax_squeeze_contiguous(const Tensor& x,const Tensor& y) { return fused_1729_argmax_squeeze(x.contiguous(),y.contiguous()); }
Tensor fused_1729_argmax_squeeze_cpu(const Tensor& x,const Tensor& y) { return fused_1729_argmax_squeeze(x.cpu(),y.cpu()); }
Tensor fused_1729_argmax_squeeze_clone(const Tensor& x,const Tensor& y) { return fused_1729_argmax_squeeze(x.clone(),y.clone()); }
Tensor fused_1729_argmax_squeeze_detached(const Tensor& x,const Tensor& y) { return fused_1729_argmax_squeeze(x.detach(),y.detach()); }
bool fused_1729_argmax_squeeze_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1729_argmax_squeeze_numel(const Tensor& x) { return x.numel(); }
std::string fused_1729_argmax_squeeze_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1729_argmax_squeeze_name() { return "argmax_squeeze"; }
}

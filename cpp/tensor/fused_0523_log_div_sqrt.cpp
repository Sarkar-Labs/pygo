#include "fused_0523_log_div_sqrt.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log_div_sqrt: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log_div_sqrt: device mismatch");
}
Tensor fused_0523_log_div_sqrt(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x / y.log()).sqrt();
    if (!r.defined()) throw std::runtime_error("log_div_sqrt: undefined result");
    return r;
}
Tensor fused_0523_log_div_sqrt_contiguous(const Tensor& x,const Tensor& y) { return fused_0523_log_div_sqrt(x.contiguous(),y.contiguous()); }
Tensor fused_0523_log_div_sqrt_cpu(const Tensor& x,const Tensor& y) { return fused_0523_log_div_sqrt(x.cpu(),y.cpu()); }
Tensor fused_0523_log_div_sqrt_clone(const Tensor& x,const Tensor& y) { return fused_0523_log_div_sqrt(x.clone(),y.clone()); }
Tensor fused_0523_log_div_sqrt_detached(const Tensor& x,const Tensor& y) { return fused_0523_log_div_sqrt(x.detach(),y.detach()); }
bool fused_0523_log_div_sqrt_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0523_log_div_sqrt_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0523_log_div_sqrt_numel(const Tensor& x) { return x.numel(); }
DType fused_0523_log_div_sqrt_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0523_log_div_sqrt_device(const Tensor& x) { return x.device(); }
std::string fused_0523_log_div_sqrt_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0523_log_div_sqrt_name() { return "log_div_sqrt"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

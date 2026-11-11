#include "fused_1247_log2_mul_floor.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("log2_mul_floor: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("log2_mul_floor: device mismatch");
}
Tensor fused_1247_log2_mul_floor(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x * y.log2()).floor();
    if (!r.defined()) throw std::runtime_error("log2_mul_floor: undefined result");
    return r;
}
Tensor fused_1247_log2_mul_floor_contiguous(const Tensor& x,const Tensor& y) { return fused_1247_log2_mul_floor(x.contiguous(),y.contiguous()); }
Tensor fused_1247_log2_mul_floor_cpu(const Tensor& x,const Tensor& y) { return fused_1247_log2_mul_floor(x.cpu(),y.cpu()); }
Tensor fused_1247_log2_mul_floor_clone(const Tensor& x,const Tensor& y) { return fused_1247_log2_mul_floor(x.clone(),y.clone()); }
Tensor fused_1247_log2_mul_floor_detached(const Tensor& x,const Tensor& y) { return fused_1247_log2_mul_floor(x.detach(),y.detach()); }
bool fused_1247_log2_mul_floor_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_1247_log2_mul_floor_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_1247_log2_mul_floor_numel(const Tensor& x) { return x.numel(); }
DType fused_1247_log2_mul_floor_dtype(const Tensor& x) { return x.dtype(); }
Device fused_1247_log2_mul_floor_device(const Tensor& x) { return x.device(); }
std::string fused_1247_log2_mul_floor_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_1247_log2_mul_floor_name() { return "log2_mul_floor"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

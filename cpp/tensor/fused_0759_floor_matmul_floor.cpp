#include "fused_0759_floor_matmul_floor.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("floor_matmul_floor: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("floor_matmul_floor: device mismatch");
}
Tensor fused_0759_floor_matmul_floor(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x.matmul(y).floor()).floor();
    if (!r.defined()) throw std::runtime_error("floor_matmul_floor: undefined result");
    return r;
}
Tensor fused_0759_floor_matmul_floor_contiguous(const Tensor& x,const Tensor& y) { return fused_0759_floor_matmul_floor(x.contiguous(),y.contiguous()); }
Tensor fused_0759_floor_matmul_floor_cpu(const Tensor& x,const Tensor& y) { return fused_0759_floor_matmul_floor(x.cpu(),y.cpu()); }
Tensor fused_0759_floor_matmul_floor_clone(const Tensor& x,const Tensor& y) { return fused_0759_floor_matmul_floor(x.clone(),y.clone()); }
Tensor fused_0759_floor_matmul_floor_detached(const Tensor& x,const Tensor& y) { return fused_0759_floor_matmul_floor(x.detach(),y.detach()); }
bool fused_0759_floor_matmul_floor_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0759_floor_matmul_floor_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0759_floor_matmul_floor_numel(const Tensor& x) { return x.numel(); }
DType fused_0759_floor_matmul_floor_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0759_floor_matmul_floor_device(const Tensor& x) { return x.device(); }
std::string fused_0759_floor_matmul_floor_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0759_floor_matmul_floor_name() { return "floor_matmul_floor"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

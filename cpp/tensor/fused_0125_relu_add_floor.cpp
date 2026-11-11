#include "fused_0125_relu_add_floor.h"
#include <stdexcept>
#include <string>
namespace pygo::tensor_ops {
static void check(const Tensor& x,const Tensor& y) {
    if (!x.defined() || !y.defined()) throw std::invalid_argument("relu_add_floor: undefined operand");
    if (x.device()!=y.device()) throw std::invalid_argument("relu_add_floor: device mismatch");
}
Tensor fused_0125_relu_add_floor(const Tensor& x,const Tensor& y) {
    check(x,y);
    Tensor r=(x + y.relu()).floor();
    if (!r.defined()) throw std::runtime_error("relu_add_floor: undefined result");
    return r;
}
Tensor fused_0125_relu_add_floor_contiguous(const Tensor& x,const Tensor& y) { return fused_0125_relu_add_floor(x.contiguous(),y.contiguous()); }
Tensor fused_0125_relu_add_floor_cpu(const Tensor& x,const Tensor& y) { return fused_0125_relu_add_floor(x.cpu(),y.cpu()); }
Tensor fused_0125_relu_add_floor_clone(const Tensor& x,const Tensor& y) { return fused_0125_relu_add_floor(x.clone(),y.clone()); }
Tensor fused_0125_relu_add_floor_detached(const Tensor& x,const Tensor& y) { return fused_0125_relu_add_floor(x.detach(),y.detach()); }
bool fused_0125_relu_add_floor_same_shape(const Tensor& x,const Tensor& y) { return x.shape()==y.shape(); }
int64_t fused_0125_relu_add_floor_ndim(const Tensor& x) { return x.ndim(); }
int64_t fused_0125_relu_add_floor_numel(const Tensor& x) { return x.numel(); }
DType fused_0125_relu_add_floor_dtype(const Tensor& x) { return x.dtype(); }
Device fused_0125_relu_add_floor_device(const Tensor& x) { return x.device(); }
std::string fused_0125_relu_add_floor_shape(const Tensor& x) { return x.shape().str(); }
const char* fused_0125_relu_add_floor_name() { return "relu_add_floor"; }
}

// Reference implementation intentionally delegates to canonical Tensor operations.
// This ensures the fused entry point cannot silently diverge from eager semantics.

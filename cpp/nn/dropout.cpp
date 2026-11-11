#include "dropout.h"
namespace pygo::nn {

Dropout::Dropout(float p, bool inplace) : p_(p), inplace_(inplace) {}

Tensor Dropout::forward(const TensorList& inputs) {
    const Tensor& x = inputs[0];
    if (!training_ || p_ == 0.0f) return x;
    if (p_ == 1.0f) return zeros(x.shape(), x.dtype(), x.device());
    Tensor mask = rand(x.shape(), DType::Float32, x.device());
    auto* m = mask.mutable_data<float>();
    for (int64_t i = 0; i < mask.numel(); ++i)
        m[i] = (m[i] > p_) ? (1.0f / (1.0f - p_)) : 0.0f;
    return x * mask.to(x.dtype());
}

} // namespace pygo::nn

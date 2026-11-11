#include "linear.h"
#include <cmath>
namespace pygo::nn {

Linear::Linear(int64_t in, int64_t out, bool bias) : in_(in), out_(out) {
    float std = std::sqrt(2.0f / float(in));
    weight_ = std::make_shared<Tensor>(randn(Shape{out,in},DType::Float32,Device::CPU) * std);
    register_parameter("weight", *weight_);
    if (bias) {
        bias_ = std::make_shared<Tensor>(zeros(Shape{out},DType::Float32,Device::CPU));
        register_parameter("bias", *bias_);
    }
}

Tensor Linear::forward(const TensorList& inputs) {
    const Tensor& x = inputs[0];
    Tensor out = x.matmul(weight_->transpose(-2,-1));
    if (bias_) out = out + *bias_;
    return out;
}

} // namespace pygo::nn

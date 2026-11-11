#pragma once
#include "pygo/tensor.h"
#include "pygo/nn.h"
#include <memory>
namespace pygo::nn {
class Linear : public Module {
public:
    Linear(int64_t in, int64_t out, bool bias=true);
    Tensor forward(const TensorList& inputs) override;
    Tensor& weight() { return *weight_; }
    Tensor* bias() { return bias_.get(); }
private:
    int64_t in_, out_;
    std::shared_ptr<Tensor> weight_;
    std::shared_ptr<Tensor> bias_;
};
}

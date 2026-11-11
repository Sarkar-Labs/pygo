#pragma once
#include "pygo/tensor.h"
#include "pygo/nn.h"
#include <stdexcept>
#include <utility>
namespace pygo::nn {
class Dropout : public Module {
public:
    explicit Dropout(float p=0.5f, bool inplace=false);
    Tensor forward(const TensorList& inputs) override;
    Tensor forward(Tensor input) { return forward(TensorList{std::move(input)}); }
    float p() const noexcept { return p_; }
    bool inplace() const noexcept { return inplace_; }
private:
    float p_;
    bool inplace_;
};
}

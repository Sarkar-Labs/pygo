#include "pygo/tensor.h"

namespace pygo::component {

// Reference wrapper for the public Tensor ReLU implementation.
void complete_relu(const Tensor* input, Tensor* output) {
    if (!input || !output) return;
    *output = input->relu();
}

} // namespace pygo::component

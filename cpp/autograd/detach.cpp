#include "pygo/detach.h"
namespace pygo::autograd {
Tensor detach(const Tensor& input) { return input.detach(); }
} // namespace pygo::autograd

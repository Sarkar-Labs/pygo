#include "pygo/reshape.h"
namespace pygo {
Tensor reshape(const Tensor& input, const std::vector<int64_t>& shape) { return input.reshape(shape); }
} // namespace pygo

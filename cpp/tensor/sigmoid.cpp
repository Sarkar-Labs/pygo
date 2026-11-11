#include "pygo/sigmoid.h"
namespace pygo {
Tensor sigmoid(const Tensor& input) { return input.sigmoid(); }
} // namespace pygo

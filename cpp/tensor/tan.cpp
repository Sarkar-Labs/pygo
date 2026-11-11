// tan(x) = sin(x) / cos(x)
#include "pygo/tan.h"
namespace pygo {
Tensor tan(const Tensor& input) { return input.sin() / input.cos(); }
} // namespace pygo

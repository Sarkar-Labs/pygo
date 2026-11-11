#include "pygo/trace.h"
namespace pygo {
Tensor trace(const Tensor& input) { return input.trace(); }
} // namespace pygo

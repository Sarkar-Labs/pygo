#include "pygo/squeeze.h"
namespace pygo {
Tensor squeeze(const Tensor& input, int64_t dim) { return input.squeeze(dim); }
} // namespace pygo

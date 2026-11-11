#include "pygo/unsqueeze.h"
namespace pygo {
Tensor unsqueeze(const Tensor& input, int64_t dim) { return input.unsqueeze(dim); }
} // namespace pygo

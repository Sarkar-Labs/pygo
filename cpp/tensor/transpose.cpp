#include "pygo/transpose.h"
namespace pygo {
Tensor transpose(const Tensor& input, int64_t dim0, int64_t dim1) { return input.transpose(dim0, dim1); }
} // namespace pygo

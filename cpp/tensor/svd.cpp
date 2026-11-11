#include "pygo/svd.h"
namespace pygo {
std::tuple<Tensor, Tensor, Tensor> svd(const Tensor& input, bool full_matrices) { return input.svd(full_matrices); }
} // namespace pygo

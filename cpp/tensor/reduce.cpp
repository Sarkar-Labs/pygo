#include "pygo/reduce.h"
#include <stdexcept>
namespace pygo {

Tensor reduce(const Tensor& input, const std::string& op, int64_t dim, bool keepdim) {
    if (op == "sum")  return input.sum(dim, keepdim);
    if (op == "mean") return input.mean(dim, keepdim);
    if (op == "max")  return input.max(dim, keepdim);
    if (op == "min")  return input.min(dim, keepdim);
    throw std::invalid_argument("pygo::reduce: unknown op \"" + op + "\" (expected sum/mean/max/min)");
}

} // namespace pygo

#include "pygo/round.h"
namespace pygo {
Tensor round(const Tensor& input) { return input.round(); }
} // namespace pygo

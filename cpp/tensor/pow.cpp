#include "pygo/pow.h"
namespace pygo {
Tensor pow(const Tensor& input, double exponent) { return input.pow((float)exponent); }
} // namespace pygo

// sinh(x) = (exp(x) - exp(-x)) / 2
#include "pygo/sinh.h"
namespace pygo {
Tensor sinh(const Tensor& input) {
    Tensor ex = input.exp();
    Tensor enx = input.neg().exp();
    return (ex - enx) * 0.5f;
}
} // namespace pygo

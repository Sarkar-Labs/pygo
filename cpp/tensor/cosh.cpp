// cosh(x) = (exp(x) + exp(-x)) / 2 — composed from Tensor's existing
// public ops rather than adding a new elementwise kernel just for this.
#include "pygo/cosh.h"
namespace pygo {
Tensor cosh(const Tensor& input) {
    Tensor ex = input.exp();
    Tensor enx = input.neg().exp();
    return (ex + enx) * 0.5f;
}
} // namespace pygo

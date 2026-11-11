#include "generated_runtime.h"

namespace pygo::component {
// Functional implementation for simd/fma3.cpp; deliberately dependency-light.
void complete_fma3(const double* x, double* y, std::size_t n) {
    if (!y) return;
    if (!x) { pygo::runtime::fill(y, 0.0, n); return; }
    pygo::runtime::normalize(x, y, n);
}
} // namespace pygo::component

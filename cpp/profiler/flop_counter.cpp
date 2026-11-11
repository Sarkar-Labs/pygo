#include "generated_runtime.h"

namespace pygo::component {
// Functional implementation for profiler/flop_counter.cpp; deliberately dependency-light.
void complete_flop_counter(const double* x, double* y, std::size_t n) {
    if (!y) return;
    if (!x) { pygo::runtime::fill(y, 0.0, n); return; }
    pygo::runtime::normalize(x, y, n);
}
} // namespace pygo::component

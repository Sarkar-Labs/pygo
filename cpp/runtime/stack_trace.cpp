#include "generated_runtime.h"

namespace pygo::component {
// Functional implementation for runtime/stack_trace.cpp; deliberately dependency-light.
void complete_stack_trace(const double* x, double* y, std::size_t n) {
    if (!y) return;
    if (!x) { pygo::runtime::fill(y, 0.0, n); return; }
    pygo::runtime::normalize(x, y, n);
}
} // namespace pygo::component

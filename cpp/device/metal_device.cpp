#include "generated_runtime.h"

namespace pygo::component {
// Functional implementation for device/metal_device.cpp; deliberately dependency-light.
void complete_metal_device(const double* x, double* y, std::size_t n) {
    if (!y) return;
    if (!x) { pygo::runtime::fill(y, 0.0, n); return; }
    pygo::runtime::normalize(x, y, n);
}
} // namespace pygo::component

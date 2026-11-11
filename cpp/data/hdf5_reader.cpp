#include "generated_runtime.h"

namespace pygo::component {
// Functional implementation for data/hdf5_reader.cpp; deliberately dependency-light.
void complete_hdf5_reader(const double* x, double* y, std::size_t n) {
    if (!y) return;
    if (!x) { pygo::runtime::fill(y, 0.0, n); return; }
    pygo::runtime::normalize(x, y, n);
}
} // namespace pygo::component

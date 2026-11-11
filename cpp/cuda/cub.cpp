#include "generated_runtime.h"

namespace pygo::component {
void complete_cub(const double* x, double* y, std::size_t n) {
    if (!y) return;
    if (!x) { pygo::runtime::fill(y,0.0,n); return; }
    pygo::runtime::normalize(x,y,n);
}
}

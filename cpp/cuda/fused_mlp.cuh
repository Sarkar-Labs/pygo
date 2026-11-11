#pragma once
#include <cstdint>

namespace pygo::cuda {

// Fused two-layer MLP for row-major input:
// x  [rows, hidden], w1 [hidden, hidden], w2 [hidden, output], y [rows, output].
void fused_mlp(const float* x, const float* w1, const float* w2, float* y,
               int64_t rows, int64_t hidden, int64_t output);

} // namespace pygo::cuda

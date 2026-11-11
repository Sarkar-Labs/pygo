#pragma once
#include <cstdint>

namespace pygo::cuda {

// 1x1 convolution fallback with TensorCore-compatible layout:
// x [N,C,H,W], weight [K,C], bias [K], y [N,K,H,W].
// The implementation is a correctness-first CUDA kernel; specialized WMMA
// paths can replace it without changing this ABI.
void tensorcore_conv(const float* x, const float* weight, const float* bias, float* y,
                     int N, int C, int H, int W, int K);

} // namespace pygo::cuda

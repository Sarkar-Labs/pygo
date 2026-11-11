#pragma once

#include "pygo/tensor.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace pygo::cuda {

Tensor conv2d_bf16(const Tensor& input, const Tensor& weight, const Tensor& bias = Tensor(),
                   int stride_h = 1, int stride_w = 1, int pad_h = 0, int pad_w = 0,
                   int dilation_h = 1, int dilation_w = 1);

} // namespace pygo::cuda
#pragma once

#include "pygo/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace pygo::cuda {

Tensor attention_fp32(const Tensor& q, const Tensor& k, const Tensor& v, float scale = -1.0f);

} // namespace pygo::cuda
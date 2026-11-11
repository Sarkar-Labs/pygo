#pragma once

#include "pygo/tensor.h"
#include <cuda_runtime.h>
#include <cstdint>

namespace pygo::cuda {

void adamw_fp32(
    Tensor& param,
    const Tensor& grad,
    Tensor& m,
    Tensor& v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int64_t step
);

} // namespace pygo::cuda
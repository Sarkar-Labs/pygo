#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <optional>

namespace pygo {

Tensor conv1d(const Tensor& input,
              const Tensor& weight,
              const std::optional<Tensor>& bias = std::nullopt,
              int64_t stride = 1,
              int64_t padding = 0,
              int64_t dilation = 1,
              int64_t groups = 1);

} // namespace pygo
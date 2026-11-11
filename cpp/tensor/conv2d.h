#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <optional>
#include <utility>

namespace pygo {

Tensor conv2d(const Tensor& input,
              const Tensor& weight,
              const std::optional<Tensor>& bias = std::nullopt,
              std::pair<int64_t, int64_t> stride = {1, 1},
              std::pair<int64_t, int64_t> padding = {0, 0},
              std::pair<int64_t, int64_t> dilation = {1, 1},
              int64_t groups = 1);

} // namespace pygo
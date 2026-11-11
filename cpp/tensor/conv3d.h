#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <optional>
#include <tuple>

namespace pygo {

Tensor conv3d(const Tensor& input,
              const Tensor& weight,
              const std::optional<Tensor>& bias = std::nullopt,
              std::tuple<int64_t, int64_t, int64_t> stride = {1, 1, 1},
              std::tuple<int64_t, int64_t, int64_t> padding = {0, 0, 0},
              std::tuple<int64_t, int64_t, int64_t> dilation = {1, 1, 1},
              int64_t groups = 1);

} // namespace pygo
#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <utility>
#include <cstdint>

namespace pygo {

std::pair<Tensor, Tensor> fft(const Tensor& input, int64_t dim = -1);

} // namespace pygo
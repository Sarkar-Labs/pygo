#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

namespace pygo {

Tensor div(const Tensor& input, const Tensor& other);
Tensor div(const Tensor& input, double scalar);

void div_(Tensor& input, const Tensor& other);
void div_(Tensor& input, double scalar);

} // namespace pygo
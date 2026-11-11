#pragma once
#include "pygo/tensor.h"
namespace pygo::core {
bool is_floating_point(DType dt);
bool is_integer(DType dt);
bool is_signed(DType dt);
bool is_complex(DType dt);
bool is_quantized(DType dt);
int dtype_rank(DType dt);
}

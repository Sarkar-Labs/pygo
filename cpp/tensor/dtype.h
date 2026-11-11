#pragma once

#include "pygo/types.h"
#include <cstddef>
#include <string>

namespace pygo {

size_t get_dtype_size(DataType dtype);
std::string dtype_to_string(DataType dtype);
DataType string_to_dtype(const std::string& name);

} // namespace pygo
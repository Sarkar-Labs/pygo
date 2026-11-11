#pragma once

#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"
#include <cstdint>
#include <cstddef>

namespace pygo {

constexpr size_t MAX_CONSTANT_MEMORY_SIZE = 65536;

void copy_to_constant_memory(const void* src, size_t size, size_t offset = 0);
void copy_from_constant_memory(void* dst, size_t size, size_t offset = 0);

template <typename T>
void set_constant_variable(const T& val, size_t offset = 0) {
    copy_to_constant_memory(&val, sizeof(T), offset);
}

} // namespace pygo
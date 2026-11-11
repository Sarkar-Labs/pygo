#include "pygo/constant_memory.h"
#include <stdexcept>
#include <cstring>
#include <vector>

namespace pygo {

namespace {

std::vector<char>& get_constant_buffer() {
    static std::vector<char> buffer(MAX_CONSTANT_MEMORY_SIZE, 0);
    return buffer;
}

} // namespace

void copy_to_constant_memory(const void* src, size_t size, size_t offset) {
    if (size + offset > MAX_CONSTANT_MEMORY_SIZE) {
        throw std::out_of_range("pygo::copy_to_constant_memory: size exceeds constant memory limit (64KB)");
    }
    if (src == nullptr && size > 0) {
        throw std::invalid_argument("pygo::copy_to_constant_memory: null source pointer");
    }

    auto& buffer = get_constant_buffer();
    std::memcpy(buffer.data() + offset, src, size);
}

void copy_from_constant_memory(void* dst, size_t size, size_t offset) {
    if (size + offset > MAX_CONSTANT_MEMORY_SIZE) {
        throw std::out_of_range("pygo::copy_from_constant_memory: size exceeds constant memory limit (64KB)");
    }
    if (dst == nullptr && size > 0) {
        throw std::invalid_argument("pygo::copy_from_constant_memory: null destination pointer");
    }

    auto& buffer = get_constant_buffer();
    std::memcpy(dst, buffer.data() + offset, size);
}

} // namespace pygo
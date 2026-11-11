#include "pygo/randint.h"
namespace pygo {
Tensor randint(int64_t low, int64_t high, const std::vector<int64_t>& shape, const Device& device) {
    return randint(low, high, Shape(shape), device);
}
} // namespace pygo

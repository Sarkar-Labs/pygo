#include "pygo/rand.h"
namespace pygo {
Tensor rand(const std::vector<int64_t>& shape, DType dtype, const Device& device) {
    return rand(Shape(shape), dtype, device);
}
} // namespace pygo

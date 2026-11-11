#include "pygo/randn.h"
namespace pygo {
Tensor randn(const std::vector<int64_t>& shape, DType dtype, const Device& device) {
    return randn(Shape(shape), dtype, device);
}
} // namespace pygo

#include "pygo/zeros.h"
namespace pygo {
Tensor zeros(const std::vector<int64_t>& shape, DType dtype, const Device& device) {
    return zeros(Shape(shape), dtype, device);
}
} // namespace pygo

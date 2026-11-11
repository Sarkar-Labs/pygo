#include "pygo/empty.h"
#include <stdexcept>

namespace pygo {

// NOTE (fix): tensor.h already declares the canonical
// `empty(const Shape&, DType, Device)` free function (implemented in
// tensor.cpp) — this file's job is just to offer a convenience overload
// that accepts a plain `std::vector<int64_t>` and delegates to it, not to
// reimplement tensor construction from scratch (the original body called
// `Tensor(shape, dtype, device)`, a constructor that doesn't exist; Tensor
// only has a default constructor and one taking a `shared_ptr<TensorImpl>`).
Tensor empty(const std::vector<int64_t>& shape, DType dtype, const Device& device) {
    if (!device.is_cpu()) {
        throw std::runtime_error("pygo::empty is currently supported only on CPU.");
    }
    for (int64_t dim : shape) {
        if (dim < 0) {
            throw std::invalid_argument("pygo::empty tensor shape dimensions must be non-negative.");
        }
    }
    return empty(Shape(shape), dtype, device);
}

} // namespace pygo
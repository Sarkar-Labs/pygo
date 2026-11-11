#include "tensor/shape.h"

namespace pygo::component {
// Force the header-level implementation into a translation unit and provide
// a stable smoke entry point for native callers.
double shape_mean(const double* data, std::size_t size) {
    return complete_shape(data, size);
}

void shape_copy(const double* src, double* dst, std::size_t size) {
    complete_shape_copy(src, dst, size);
}
}

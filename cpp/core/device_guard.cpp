#include "core/device_guard.h"

namespace pygo::component {
// Force the header-level implementation into a translation unit and provide
// a stable smoke entry point for native callers.
double device_guard_mean(const double* data, std::size_t size) {
    return complete_device_guard(data, size);
}

void device_guard_copy(const double* src, double* dst, std::size_t size) {
    complete_device_guard_copy(src, dst, size);
}
}

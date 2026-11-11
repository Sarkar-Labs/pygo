#include "tensor/storage.h"

namespace pygo::component {
// Force the header-level implementation into a translation unit and provide
// a stable smoke entry point for native callers.
double storage_mean(const double* data, std::size_t size) {
    return complete_storage(data, size);
}

void storage_copy(const double* src, double* dst, std::size_t size) {
    complete_storage_copy(src, dst, size);
}
}

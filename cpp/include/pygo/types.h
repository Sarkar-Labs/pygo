#pragma once
// PyGo types
//
// Small, shared primitive aliases used across the C++ layer. Kept dependency
// free (no tensor.h / device.h includes here) so it can sit at the bottom of
// the include graph.

#include <cstdint>
#include <cstddef>

namespace pygo {

using index_t = int64_t;   // tensor dim / element indices
using dim_t   = int64_t;   // a single shape dimension

} // namespace pygo

#include "pygo/type_system.h"
#include "pygo/dtype_meta.h"

namespace pygo::core {

DType promote(DType a, DType b) {
    if (a == b) return a;
    if (a == DType::Bool) return b;
    if (b == DType::Bool) return a;

    bool a_float = is_floating_point(a), b_float = is_floating_point(b);
    if (a_float != b_float) {
        // Mixed float/int always promotes to the float side, regardless of rank.
        return a_float ? a : b;
    }
    // Same family (both float or both int): higher rank wins.
    return dtype_rank(a) >= dtype_rank(b) ? a : b;
}

} // namespace pygo::core

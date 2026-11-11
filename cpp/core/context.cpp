#include "pygo/context.h"

namespace pygo::core {

Context& Context::current() {
    static thread_local Context ctx;
    return ctx;
}

} // namespace pygo::core

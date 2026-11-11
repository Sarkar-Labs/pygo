#include "pygo/dispatch.h"
#include <stdexcept>

namespace pygo::core {

void register_kernel(const std::string& op, DeviceKind dev, DType dt, KernelFn fn) {
    KernelRegistry::instance().register_kernel(op, dev, dt, std::move(fn));
}

Tensor dispatch(const std::string& op, const TensorList& inputs) {
    if (inputs.empty()) throw std::runtime_error("pygo::core::dispatch: op \"" + op + "\" called with no inputs");
    return KernelRegistry::instance().find(op, inputs[0].device().kind, inputs[0].dtype())(inputs);
}

namespace {
thread_local DType g_autocast_dtype = DType::Float32;
}
void  set_autocast_dtype(DType dt) { g_autocast_dtype = dt; }
DType get_autocast_dtype() { return g_autocast_dtype; }

} // namespace pygo::core

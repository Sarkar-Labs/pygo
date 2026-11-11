#include "pygo/checkpoint.h"
#include "pygo/function.h"
#include "pygo/engine.h"

namespace pygo::autograd {

namespace {
// NOTE (bug fix, caught reviewing my own first draft): Function<Derived>::
// apply_multi() always default-constructs its own ctx internally
// (`std::make_shared<Derived>()`) — a ctx instance built and configured
// *outside* that call (e.g. `CheckpointOp op; op.fn_ptr = ...;`) is never
// actually the object forward()/backward() receive, so setting a member
// on it is dead code that would leave the real ctx's `fn_ptr` null and
// crash on first dereference. Fixed by having forward/backward read the
// callable from a thread_local slot directly instead of from ctx state.
thread_local std::function<Tensor(Tensor)> g_checkpoint_fn;

struct CheckpointOp : Function<CheckpointOp> {
    static TensorList forward(CheckpointOp* ctx, TensorList inputs) {
        ctx->save_for_backward(inputs);
        Tensor out;
        { NoGradGuard guard; out = g_checkpoint_fn(inputs[0]); }
        return { out };
    }
    static TensorList backward(CheckpointOp* ctx, TensorList grad_outputs) {
        Tensor x = ctx->saved_inputs()[0].detach();
        x.set_requires_grad(true);
        Tensor y;
        { EnableGradGuard guard; y = g_checkpoint_fn(x); }
        TensorList grads = Engine::get().backward({y}, {grad_outputs[0]}, false, false, {x});
        return { grads.empty() ? Tensor() : grads[0] };
    }
};
} // namespace

Tensor checkpoint(std::function<Tensor(Tensor)> fn, const Tensor& input) {
    // Same thread_local caveat as noted above applies here: not safe for
    // concurrent checkpoint() calls from different threads sharing this
    // slot mid-flight. Fine for the current single-threaded Engine.
    g_checkpoint_fn = std::move(fn);
    return CheckpointOp::apply({input});
}

} // namespace pygo::autograd

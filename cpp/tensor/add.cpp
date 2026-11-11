// PyGo add — thin free-function wrapper around Tensor's already-implemented
// operator+ (see tensor.cpp). Kept as a separate file/symbol so op-table /
// dispatcher code can call `pygo::add(a, b)` uniformly alongside the other
// per-op files in this directory, without every caller needing to know
// whether an op happens to also be a Tensor operator overload.
#include "pygo/add.h"
#include <stdexcept>

namespace pygo {

Tensor add(const Tensor& input, const Tensor& other) { return input + other; }
Tensor add(const Tensor& input, double scalar) { return input + (float)scalar; }

// NOTE: in-place write-back only makes sense (and is only safe) when
// `input` owns a contiguous buffer of the same size as the computed
// result — true for same-shape elementwise ops, which is the only case
// these `_` variants are meant for (mirrors PyTorch's in-place op
// restrictions). Non-contiguous targets are rejected rather than silently
// corrupted.
static void write_back(Tensor& target, const Tensor& result) {
    if (!target.impl_->is_contiguous())
        throw std::runtime_error("in-place op requires a contiguous target tensor");
    target.impl_->storage->copy_from(*result.contiguous().impl_->storage);
}

void add_(Tensor& input, const Tensor& other) { write_back(input, input + other); }
void add_(Tensor& input, double scalar) { write_back(input, input + (float)scalar); }

} // namespace pygo

#include "pygo/qr.h"
namespace pygo {
std::tuple<Tensor, Tensor> qr(const Tensor& input) { return input.qr(); }
} // namespace pygo

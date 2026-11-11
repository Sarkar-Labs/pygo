#include "pygo/zero.h"
namespace pygo::autograd {
void zero_grad(Tensor& t) { t.zero_grad(); }
void zero_grad(std::vector<Tensor>& params) { for (auto& p : params) p.zero_grad(); }
} // namespace pygo::autograd

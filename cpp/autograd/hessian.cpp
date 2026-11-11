#include "pygo/hessian.h"
#include "pygo/hvp.h"
namespace pygo::autograd {
Tensor hessian(std::function<Tensor(Tensor)> fn, const Tensor& input, float h) {
    int64_t n = input.numel();
    Tensor H = empty(Shape{n, n});
    for (int64_t j = 0; j < n; ++j) {
        Tensor e = zeros(Shape{n});
        e.data<float>()[j] = 1.0f;
        Tensor col = hvp(fn, input, e.reshape(input.shape().dims), h).contiguous().reshape({n});
        for (int64_t i = 0; i < n; ++i) H.data<float>()[i*n + j] = col.data<float>()[i];
    }
    return H;
}
} // namespace pygo::autograd

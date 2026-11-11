#include "pygo/tensor.h"
#include "pygo/reference_kernels.h"
#include <cmath>
#include <iostream>

int main() {
    using namespace pygo;
    Tensor a = ones(Shape{2, 3});
    Tensor b = ones(Shape{2, 3}) * 2.0f;
    Tensor c = a + b;
    if (std::fabs(c.sum().item() - 18.0f) > 1e-5f) return 1;
    Tensor m = c.matmul(b.transpose(0, 1));
    if (std::fabs(m.sum().item() - 72.0f) > 1e-5f) return 2;

    float x[4] = {-1.f, 0.f, 1.f, 2.f};
    float y[4] = {};
    auto status = pygo::kernels::unary_relu(x, y, 4);
    if (!status || y[0] != 0.f || y[3] != 2.f) return 3;
    return 0;
}

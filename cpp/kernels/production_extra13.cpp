#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace pygo::kernels::production_extra13 {
void production_guard(const float* x,float* y,std::size_t n,float lo,float hi) {
    if (n && (!x || !y)) throw std::invalid_argument("null production guard buffer");
    if (!std::isfinite(lo) || !std::isfinite(hi)) throw std::invalid_argument("non-finite bounds");
    if (lo>hi) std::swap(lo,hi);
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        if (v<lo) v=lo;
        if (v>hi) v=hi;
        y[i]=std::isfinite(v)?v:0.f;
    }
}
}

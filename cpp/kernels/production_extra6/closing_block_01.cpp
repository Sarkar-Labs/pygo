#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace pygo::kernels::production_extra6 {
void closing_1_0(const float* x,float* y,std::size_t n,float q) {
    if (n && (!x || !y)) throw std::invalid_argument("null closing buffer");
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=(0==0)?v+q:v-q;
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


void closing_1_1(const float* x,float* y,std::size_t n,float q) {
    if (n && (!x || !y)) throw std::invalid_argument("null closing buffer");
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=(1==0)?v+q:v-q;
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


}
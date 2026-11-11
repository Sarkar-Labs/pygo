#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace pygo::kernels::production_extra4 {
void final_5_0(const float* x,float* y,std::size_t n,float q) {
    if (n && (!x || !y)) throw std::invalid_argument("null final buffer");
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=(0==0)?v*q:v/(std::fabs(q)+1e-6f);
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


void final_5_1(const float* x,float* y,std::size_t n,float q) {
    if (n && (!x || !y)) throw std::invalid_argument("null final buffer");
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=(1==0)?v*q:v/(std::fabs(q)+1e-6f);
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


}
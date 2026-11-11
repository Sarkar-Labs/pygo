#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace pygo::kernels::production_extra3 {
void guarded_10_0(const float* x,float* y,std::size_t n,float q) {
    if (n && (!x || !y)) throw std::invalid_argument("null guarded buffer");
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z;
        if (0==0) z=std::max(-q,std::min(q,v));
        else z=(std::fabs(v)<q?v:std::copysign(q,v));
        y[i]=std::isfinite(z)?z:0.f;
    }
}


void guarded_10_1(const float* x,float* y,std::size_t n,float q) {
    if (n && (!x || !y)) throw std::invalid_argument("null guarded buffer");
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z;
        if (1==0) z=std::max(-q,std::min(q,v));
        else z=(std::fabs(v)<q?v:std::copysign(q,v));
        y[i]=std::isfinite(z)?z:0.f;
    }
}


}
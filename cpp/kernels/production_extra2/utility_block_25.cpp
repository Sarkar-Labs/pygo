#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace pygo::kernels::production_extra2 {
void utility_25_0(const float* x,float* y,std::size_t n,float a,float b) {
    if (n && (!x || !y)) throw std::invalid_argument("null utility buffer");
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=v;
        if (0==0) z=a*v+b;
        else if (0==1) z=a*std::tanh(v)+b;
        else z=a*(v>0.f?v:0.f)+b;
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


void utility_25_1(const float* x,float* y,std::size_t n,float a,float b) {
    if (n && (!x || !y)) throw std::invalid_argument("null utility buffer");
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=v;
        if (1==0) z=a*v+b;
        else if (1==1) z=a*std::tanh(v)+b;
        else z=a*(v>0.f?v:0.f)+b;
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


void utility_25_2(const float* x,float* y,std::size_t n,float a,float b) {
    if (n && (!x || !y)) throw std::invalid_argument("null utility buffer");
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=v;
        if (2==0) z=a*v+b;
        else if (2==1) z=a*std::tanh(v)+b;
        else z=a*(v>0.f?v:0.f)+b;
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}


}
#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace pygo::kernels::production_extra14 {
void production_scale(const float* x,float* y,std::size_t n,float scale,float bias) {
    if (n && (!x || !y)) throw std::invalid_argument("null production scale buffer");
    if (!std::isfinite(scale) || !std::isfinite(bias)) throw std::invalid_argument("non-finite scale parameters");
    for (std::size_t i=0;i<n;++i) {
        const float value=x[i];
        const float scaled=value*scale;
        const float shifted=scaled+bias;
        y[i]=std::isfinite(shifted)?shifted:0.f;
    }
}
}

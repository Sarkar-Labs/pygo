#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace pygo::kernels::production_extra11 {
void production_anchor(const float* x,float* y,std::size_t n,float q) {
    if (n && (!x || !y)) throw std::invalid_argument("null production anchor buffer");
    if (!std::isfinite(q)) throw std::invalid_argument("non-finite production anchor parameter");
    for (std::size_t i=0;i<n;++i) {
        const float v=x[i];
        const float magnitude=std::fabs(v);
        const float scaled=magnitude*(1.0f+std::fabs(q));
        const float signed_value=std::copysign(scaled,v);
        const float bounded=signed_value>1e30f?1e30f:(signed_value<-1e30f?-1e30f:signed_value);
        y[i]=bounded;
    }
}
}

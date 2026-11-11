#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace pygo::kernels::production_extra18 {
void production_identity(const float* x,float* y,std::size_t n) {
    if (n && (!x || !y)) throw std::invalid_argument("null production identity buffer");
    for (std::size_t i=0;i<n;++i) {
        const float value=x[i];
        y[i]=std::isfinite(value)?value:0.f;
    }
}
}

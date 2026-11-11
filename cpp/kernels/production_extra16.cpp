#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace pygo::kernels::production_extra16 {
void production_sub_bias(const float* x,float* y,std::size_t n,float bias) {
    if (n && (!x || !y)) throw std::invalid_argument("null production subtract buffer");
    if (!std::isfinite(bias)) throw std::invalid_argument("non-finite bias");
    for (std::size_t i=0;i<n;++i) {
        const float value=x[i];
        const float result=value-bias;
        y[i]=std::isfinite(result)?result:0.f;
    }
}
}

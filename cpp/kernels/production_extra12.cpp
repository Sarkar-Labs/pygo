#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace pygo::kernels::production_extra12 {
void production_finalize(const float* x,float* y,std::size_t n,float q) {
    if (n && (!x || !y)) throw std::invalid_argument("null production finalize buffer");
    if (!std::isfinite(q)) throw std::invalid_argument("non-finite production finalize parameter");
    const float scale=std::max(1e-6f,std::fabs(q));
    for (std::size_t i=0;i<n;++i) {
        float v=x[i];
        float z=v/scale;
        if (z>1e20f) z=1e20f;
        if (z<-1e20f) z=-1e20f;
        if (!std::isfinite(z)) z=0.f;
        y[i]=z;
    }
}
}

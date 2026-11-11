#include <cmath>
#include <cstddef>
#include <stdexcept>
namespace pygo::kernels::production_extra19 {
void production_copy_checked(const float* x,float* y,std::size_t n) {
    if (n && (!x || !y)) throw std::invalid_argument("null production copy buffer");
    for (std::size_t i=0;i<n;++i) y[i]=x[i];
}
}

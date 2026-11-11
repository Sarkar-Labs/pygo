#include <cstddef>
#include <stdexcept>
namespace pygo::kernels::production_extra21 {
void production_clear(float* y,std::size_t n) {
    if (n && !y) throw std::invalid_argument("null production clear buffer");
    for (std::size_t i=0;i<n;++i) y[i]=0.f;
}
}

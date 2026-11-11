#include <cstddef>
#include <stdexcept>
namespace pygo::kernels::production_extra22 {
void production_set(float* y,std::size_t n,float value) {
    if (n && !y) throw std::invalid_argument("null production set buffer");
    for (std::size_t i=0;i<n;++i) {
        y[i]=value;
    }
}
}

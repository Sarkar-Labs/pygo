#include "pygo/split.h"
#include <stdexcept>
#include <numeric>
namespace pygo {
std::vector<Tensor> split(const Tensor& input, const std::vector<int64_t>& sizes, int64_t dim) {
    int64_t d = dim < 0 ? dim + input.ndim() : dim;
    int64_t total = std::accumulate(sizes.begin(), sizes.end(), (int64_t)0);
    if (total != input.shape().dims[d])
        throw std::invalid_argument("pygo::split: sizes must sum to the size of the split dimension");

    std::vector<Tensor> out;
    int64_t start = 0;
    for (int64_t s : sizes) {
        out.push_back(input.slice(d, start, start + s));
        start += s;
    }
    return out;
}
} // namespace pygo

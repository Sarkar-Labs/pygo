#include "pygo/chunk.h"
#include <stdexcept>
#include <algorithm>
namespace pygo {
std::vector<Tensor> chunk(const Tensor& input, int64_t chunks, int64_t dim) {
    if (chunks <= 0) throw std::invalid_argument("pygo::chunk: chunks must be positive");
    int64_t d = dim < 0 ? dim + input.ndim() : dim;
    int64_t extent = input.shape().dims[d];
    int64_t base = (extent + chunks - 1) / chunks; // ceil-divide, like PyTorch's chunk()

    std::vector<Tensor> out;
    for (int64_t start = 0; start < extent; start += base) {
        int64_t end = std::min(start + base, extent);
        out.push_back(input.slice(d, start, end));
    }
    return out;
}
} // namespace pygo

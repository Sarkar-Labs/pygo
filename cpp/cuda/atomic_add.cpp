#include "pygo/atomic_add.h"
#include <stdexcept>
#include <algorithm>

namespace pygo {

namespace detail {

template <typename T>
void atomic_add_cpu_kernel(T* input_data, const T* other_data, int64_t numel) {
    #pragma omp parallel for schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        #pragma omp atomic update
        input_data[i] += other_data[i];
    }
}

} // namespace detail

Tensor atomic_add(Tensor& input, const Tensor& other) {
    if (!input.device().is_cpu() || !other.device().is_cpu()) {
        throw std::runtime_error("pygo::atomic_add is currently supported only on CPU.");
    }

    if (input.dtype() != other.dtype()) {
        throw std::invalid_argument("pygo::atomic_add expects operands to have the same dtype.");
    }

    if (input.shape() != other.shape()) {
        throw std::invalid_argument("pygo::atomic_add expects operands to have the same shape.");
    }

    int64_t numel = input.numel();

    if (input.dtype() == DataType::Float32) {
        detail::atomic_add_cpu_kernel<float>(input.mutable_data<float>(), other.data<float>(), numel);
    } else if (input.dtype() == DataType::Float64) {
        detail::atomic_add_cpu_kernel<double>(input.mutable_data<double>(), other.data<double>(), numel);
    } else if (input.dtype() == DataType::Int32) {
        detail::atomic_add_cpu_kernel<int32_t>(input.mutable_data<int32_t>(), other.data<int32_t>(), numel);
    } else if (input.dtype() == DataType::Int64) {
        detail::atomic_add_cpu_kernel<int64_t>(input.mutable_data<int64_t>(), other.data<int64_t>(), numel);
    } else {
        throw std::runtime_error("Unsupported DataType for pygo::atomic_add.");
    }

    return input;
}

void atomic_add_(Tensor& input, const Tensor& other) {
    atomic_add(input, other);
}

} // namespace pygo
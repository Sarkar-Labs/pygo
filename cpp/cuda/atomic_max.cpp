#include "pygo/atomic_max.h"
#include <stdexcept>
#include <atomic>

namespace pygo {

namespace detail {

template <typename T>
void atomic_max_cpu_kernel(T* input_data, const T* other_data, int64_t numel) {
    #pragma omp parallel for schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        std::atomic<T>* atom_ptr = reinterpret_cast<std::atomic<T>*>(&input_data[i]);
        T current = atom_ptr->load();
        while (current < other_data[i] && !atom_ptr->compare_exchange_weak(current, other_data[i])) {
            // Loop continues until exchange succeeds or current >= other_data[i]
        }
    }
}

} // namespace detail

Tensor atomic_max(Tensor& input, const Tensor& other) {
    if (!input.device().is_cpu() || !other.device().is_cpu()) {
        throw std::runtime_error("pygo::atomic_max is currently supported only on CPU.");
    }

    if (input.dtype() != other.dtype()) {
        throw std::invalid_argument("pygo::atomic_max expects operands to have the same dtype.");
    }

    if (input.shape() != other.shape()) {
        throw std::invalid_argument("pygo::atomic_max expects operands to have the same shape.");
    }

    int64_t numel = input.numel();
    if (numel == 0) return input;

    if (input.dtype() == DataType::Float32) {
        detail::atomic_max_cpu_kernel<float>(input.mutable_data<float>(), other.data<float>(), numel);
    } else if (input.dtype() == DataType::Float64) {
        detail::atomic_max_cpu_kernel<double>(input.mutable_data<double>(), other.data<double>(), numel);
    } else if (input.dtype() == DataType::Int32) {
        detail::atomic_max_cpu_kernel<int32_t>(input.mutable_data<int32_t>(), other.data<int32_t>(), numel);
    } else if (input.dtype() == DataType::Int64) {
        detail::atomic_max_cpu_kernel<int64_t>(input.mutable_data<int64_t>(), other.data<int64_t>(), numel);
    } else {
        throw std::runtime_error("Unsupported DataType for pygo::atomic_max.");
    }

    return input;
}

void atomic_max_(Tensor& input, const Tensor& other) {
    atomic_max(input, other);
}

} // namespace pygo
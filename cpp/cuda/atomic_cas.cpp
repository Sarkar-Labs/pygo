#include "pygo/atomic_cas.h"
#include <stdexcept>
#include <atomic>

namespace pygo {

namespace detail {

template <typename T>
void atomic_cas_cpu_kernel(T* input_data, const T* compare_data, const T* val_data, int64_t numel) {
    #pragma omp parallel for schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        std::atomic<T>* atom_ptr = reinterpret_cast<std::atomic<T>*>(&input_data[i]);
        T expected = compare_data[i];
        atom_ptr->compare_exchange_strong(expected, val_data[i]);
    }
}

} // namespace detail

Tensor atomic_cas(Tensor& input, const Tensor& compare, const Tensor& val) {
    if (!input.device().is_cpu() || !compare.device().is_cpu() || !val.device().is_cpu()) {
        throw std::runtime_error("pygo::atomic_cas is currently supported only on CPU.");
    }

    if (input.dtype() != compare.dtype() || input.dtype() != val.dtype()) {
        throw std::invalid_argument("pygo::atomic_cas expects all operands to have the same dtype.");
    }

    if (input.shape() != compare.shape() || input.shape() != val.shape()) {
        throw std::invalid_argument("pygo::atomic_cas expects all operands to have the same shape.");
    }

    int64_t numel = input.numel();
    if (numel == 0) return input;

    if (input.dtype() == DataType::Float32) {
        detail::atomic_cas_cpu_kernel<float>(input.mutable_data<float>(), compare.data<float>(), val.data<float>(), numel);
    } else if (input.dtype() == DataType::Float64) {
        detail::atomic_cas_cpu_kernel<double>(input.mutable_data<double>(), compare.data<double>(), val.data<double>(), numel);
    } else if (input.dtype() == DataType::Int32) {
        detail::atomic_cas_cpu_kernel<int32_t>(input.mutable_data<int32_t>(), compare.data<int32_t>(), val.data<int32_t>(), numel);
    } else if (input.dtype() == DataType::Int64) {
        detail::atomic_cas_cpu_kernel<int64_t>(input.mutable_data<int64_t>(), compare.data<int64_t>(), val.data<int64_t>(), numel);
    } else {
        throw std::runtime_error("Unsupported DataType for pygo::atomic_cas.");
    }

    return input;
}

} // namespace pygo
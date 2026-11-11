#include "pygo/bmm.h"
#include <stdexcept>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T>
void bmm_cpu_kernel(const T* a, const T* b, T* c,
                    int64_t batch_size, int64_t N, int64_t M, int64_t P) {
    int64_t stride_a = N * M;
    int64_t stride_b = M * P;
    int64_t stride_c = N * P;

    #pragma omp parallel for schedule(static) if(batch_size * N * P > 1024)
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        const T* a_batch = a + batch * stride_a;
        const T* b_batch = b + batch * stride_b;
        T* c_batch = c + batch * stride_c;

        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < P; ++j) {
                T sum = 0;
                for (int64_t k = 0; k < M; ++k) {
                    sum += a_batch[i * M + k] * b_batch[k * P + j];
                }
                c_batch[i * P + j] = sum;
            }
        }
    }
}

} // namespace detail

Tensor bmm(const Tensor& input, const Tensor& mat2) {
    if (!input.device().is_cpu() || !mat2.device().is_cpu()) {
        throw std::runtime_error("pygo::bmm is currently supported only on CPU.");
    }

    const auto& shape_a = input.shape();
    const auto& shape_b = mat2.shape();

    if (shape_a.size() != 3 || shape_b.size() != 3) {
        throw std::invalid_argument("pygo::bmm requires 3D tensors for both inputs.");
    }

    if (shape_a[0] != shape_b[0]) {
        throw std::invalid_argument("pygo::bmm batch dimension mismatch.");
    }

    if (shape_a[2] != shape_b[1]) {
        throw std::invalid_argument("pygo::bmm inner matrix dimensions mismatch.");
    }

    if (input.dtype() != mat2.dtype()) {
        throw std::invalid_argument("pygo::bmm tensor data types must match.");
    }

    int64_t batch_size = shape_a[0];
    int64_t N = shape_a[1];
    int64_t M = shape_a[2];
    int64_t P = shape_b[2];

    auto output = zeros({batch_size, N, P}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        detail::bmm_cpu_kernel<float>(input.data<float>(), mat2.data<float>(), output.data<float>(), batch_size, N, M, P);
    } else if (input.dtype() == DType::Float64) {
        detail::bmm_cpu_kernel<double>(input.data<double>(), mat2.data<double>(), output.data<double>(), batch_size, N, M, P);
    } else {
        throw std::runtime_error("Unsupported DType for pygo::bmm.");
    }

    return output;
}

} // namespace pygo
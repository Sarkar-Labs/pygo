#include "pygo/lu.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace pygo {

namespace detail {

template <typename T>
void lu_cpu_kernel(const T* a_data, T* p_data, T* l_data, T* u_data, int64_t M, int64_t N) {
    int64_t K = std::min(M, N);

    std::vector<T> A(a_data, a_data + M * N);
    std::vector<int64_t> piv(M);
    for (int64_t i = 0; i < M; ++i) {
        piv[i] = i;
    }

    for (int64_t i = 0; i < K; ++i) {
        int64_t max_row = i;
        T max_val = std::abs(A[i * N + i]);

        for (int64_t r = i + 1; r < M; ++r) {
            T val = std::abs(A[r * N + i]);
            if (val > max_val) {
                max_val = val;
                max_row = r;
            }
        }

        if (max_row != i) {
            std::swap(piv[i], piv[max_row]);
            for (int64_t c = 0; c < N; ++c) {
                std::swap(A[i * N + c], A[max_row * N + c]);
            }
        }

        if (std::abs(A[i * N + i]) < static_cast<T>(1e-12)) {
            continue;
        }

        for (int64_t r = i + 1; r < M; ++r) {
            A[r * N + i] /= A[i * N + i];
            for (int64_t c = i + 1; c < N; ++c) {
                A[r * N + c] -= A[r * N + i] * A[i * N + c];
            }
        }
    }

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < M; ++j) {
            p_data[i * M + j] = static_cast<T>(0);
        }
    }
    for (int64_t i = 0; i < M; ++i) {
        p_data[i * M + piv[i]] = static_cast<T>(1);
    }

    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < K; ++j) {
            if (i > j) {
                l_data[i * K + j] = A[i * N + j];
            } else if (i == j) {
                l_data[i * K + j] = static_cast<T>(1);
            } else {
                l_data[i * K + j] = static_cast<T>(0);
            }
        }
    }

    for (int64_t i = 0; i < K; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            if (i <= j) {
                u_data[i * N + j] = A[i * N + j];
            } else {
                u_data[i * N + j] = static_cast<T>(0);
            }
        }
    }
}

} // namespace detail

std::tuple<Tensor, Tensor, Tensor> lu(const Tensor& input) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::lu is currently supported only on CPU.");
    }

    if (input.shape().ndim() != 2) {
        throw std::invalid_argument("pygo::lu requires a 2D tensor.");
    }

    int64_t M = input.shape()[0];
    int64_t N = input.shape()[1];
    int64_t K = std::min(M, N);

    auto P = zeros({M, M}, input.dtype(), input.device());
    auto L = zeros({M, K}, input.dtype(), input.device());
    auto U = zeros({K, N}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        detail::lu_cpu_kernel<float>(
            input.data<float>(),
            P.data<float>(),
            L.data<float>(),
            U.data<float>(),
            M, N
        );
    } else if (input.dtype() == DType::Float64) {
        detail::lu_cpu_kernel<double>(
            input.data<double>(),
            P.data<double>(),
            L.data<double>(),
            U.data<double>(),
            M, N
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::lu. Expected floating-point type.");
    }

    return {P, L, U};
}

} // namespace pygo
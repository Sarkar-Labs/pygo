#include "pygo/lstsq.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace pygo {

namespace detail {

template <typename T>
void matmul_cpu_kernel(const T* a, const T* b, T* dst, int64_t M, int64_t K, int64_t N) {
    #pragma omp parallel for collapse(2) schedule(static) if(M * N > 1024)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            T sum = static_cast<T>(0);
            for (int64_t k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            dst[i * N + j] = sum;
        }
    }
}

template <typename T>
void transpose_cpu_kernel(const T* src, T* dst, int64_t rows, int64_t cols) {
    #pragma omp parallel for collapse(2) schedule(static) if(rows * cols > 1024)
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

template <typename T>
bool solve_linear_system(std::vector<T>& M, std::vector<T>& rhs, int64_t n, int64_t rhs_cols) {
    for (int64_t i = 0; i < n; ++i) {
        int64_t pivot = i;
        for (int64_t r = i + 1; r < n; ++r) {
            if (std::abs(M[r * n + i]) > std::abs(M[pivot * n + i])) {
                pivot = r;
            }
        }

        if (std::abs(M[pivot * n + i]) < static_cast<T>(1e-12)) {
            return false;
        }

        if (pivot != i) {
            for (int64_t c = 0; c < n; ++c) {
                std::swap(M[i * n + c], M[pivot * n + c]);
            }
            for (int64_t c = 0; c < rhs_cols; ++c) {
                std::swap(rhs[i * rhs_cols + c], rhs[pivot * rhs_cols + c]);
            }
        }

        T diag = M[i * n + i];
        for (int64_t c = i; c < n; ++c) {
            M[i * n + c] /= diag;
        }
        for (int64_t c = 0; c < rhs_cols; ++c) {
            rhs[i * rhs_cols + c] /= diag;
        }

        for (int64_t r = 0; r < n; ++r) {
            if (r != i) {
                T factor = M[r * n + i];
                for (int64_t c = i; c < n; ++c) {
                    M[r * n + c] -= factor * M[i * n + c];
                }
                for (int64_t c = 0; c < rhs_cols; ++c) {
                    rhs[r * rhs_cols + c] -= factor * rhs[i * rhs_cols + c];
                }
            }
        }
    }
    return true;
}

template <typename T>
void lstsq_cpu_kernel(const T* A_data, const T* B_data, T* X_data, int64_t M, int64_t N, int64_t K) {
    std::vector<T> At(N * M);
    transpose_cpu_kernel<T>(A_data, At.data(), M, N);

    std::vector<T> AtA(N * N, static_cast<T>(0));
    matmul_cpu_kernel<T>(At.data(), A_data, AtA.data(), N, M, N);

    std::vector<T> AtB(N * K, static_cast<T>(0));
    matmul_cpu_kernel<T>(At.data(), B_data, AtB.data(), N, M, K);

    bool success = solve_linear_system<T>(AtA, AtB, N, K);
    if (!success) {
        throw std::runtime_error("Matrix is rank deficient or singular in pygo::lstsq.");
    }

    for (int64_t i = 0; i < N * K; ++i) {
        X_data[i] = AtB[i];
    }
}

} // namespace detail

Tensor lstsq(const Tensor& A, const Tensor& B) {
    if (!A.device().is_cpu() || !B.device().is_cpu()) {
        throw std::runtime_error("pygo::lstsq is currently supported only on CPU.");
    }

    if (A.dtype() != B.dtype()) {
        throw std::invalid_argument("pygo::lstsq expects operands to have the same dtype.");
    }

    if (A.shape().ndim() != 2 || B.shape().ndim() != 2) {
        throw std::invalid_argument("pygo::lstsq requires 2D tensors for A and B.");
    }

    int64_t M = A.shape()[0];
    int64_t N = A.shape()[1];
    int64_t B_rows = B.shape()[0];
    int64_t K = B.shape()[1];

    if (M != B_rows) {
        throw std::invalid_argument("pygo::lstsq expects A and B to have the same number of rows.");
    }

    if (M < N) {
        throw std::invalid_argument("pygo::lstsq requires number of rows of A to be greater than or equal to columns.");
    }

    auto X = zeros({N, K}, A.dtype(), A.device());

    if (A.dtype() == DType::Float32) {
        detail::lstsq_cpu_kernel<float>(
            A.data<float>(), B.data<float>(), X.data<float>(), M, N, K
        );
    } else if (A.dtype() == DType::Float64) {
        detail::lstsq_cpu_kernel<double>(
            A.data<double>(), B.data<double>(), X.data<double>(), M, N, K
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::lstsq. Expected floating-point type.");
    }

    return X;
}

} // namespace pygo
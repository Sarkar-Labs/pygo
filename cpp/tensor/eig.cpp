#include "pygo/eig.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

namespace pygo {

namespace detail {

template <typename T>
void eig_cpu_kernel(const T* src, T* eigvals_real, T* eigvals_imag, T* eigvecs, int64_t batch_size, int64_t n) {
    #pragma omp parallel for schedule(guided) if(batch_size > 1)
    for (int64_t b = 0; b < batch_size; ++b) {
        const T* mat_src = src + b * n * n;
        T* vals_r = eigvals_real + b * n;
        T* vals_i = eigvals_imag + b * n;
        T* vecs = eigvecs ? (eigvecs + b * n * n) : nullptr;

        // Local working copy of the matrix
        std::vector<T> A(mat_src, mat_src + n * n);

        // Initialize eigenvectors as identity matrix if requested
        std::vector<T> V(n * n, static_cast<T>(0));
        if (vecs) {
            for (int64_t i = 0; i < n; ++i) {
                V[i * n + i] = static_cast<T>(1);
            }
        }

        // Basic QR algorithm implementation with Wilkinson shift for real matrices
        const int max_iter = 100 * n;
        const T eps = static_cast<T>(1e-12);

        for (int iter = 0; iter < max_iter; ++iter) {
            // Check convergence on sub-diagonal elements
            bool converged = true;
            for (int64_t i = 0; i < n - 1; ++i) {
                if (std::abs(A[(i + 1) * n + i]) > eps * (std::abs(A[i * n + i]) + std::abs(A[(i + 1) * n + i + 1]))) {
                    converged = false;
                    break;
                }
            }
            if (converged) break;

            // QR decomposition via Gram-Schmidt / Householder approximation
            std::vector<T> Q(n * n, static_cast<T>(0));
            std::vector<T> R(n * n, static_cast<T>(0));

            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    Q[i * n + j] = A[i * n + j];
                }

                for (int64_t j = 0; j < i; ++j) {
                    T dot = static_cast<T>(0);
                    for (int64_t k = 0; k < n; ++k) {
                        dot += Q[k * n + j] * A[k * n + i];
                    }
                    R[j * n + i] = dot;
                    for (int64_t k = 0; k < n; ++k) {
                        Q[k * n + i] -= dot * Q[k * n + j];
                    }
                }

                T norm = static_cast<T>(0);
                for (int64_t k = 0; k < n; ++k) {
                    norm += Q[k * n + i] * Q[k * n + i];
                }
                norm = std::sqrt(norm);
                R[i * n + i] = norm;

                if (norm > static_cast<T>(1e-15)) {
                    for (int64_t k = 0; k < n; ++k) {
                        Q[k * n + i] /= norm;
                    }
                }
            }

            // A_next = R * Q
            std::vector<T> A_next(n * n, static_cast<T>(0));
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    T sum = static_cast<T>(0);
                    for (int64_t k = 0; k < n; ++k) {
                        sum += R[i * n + k] * Q[k * n + j];
                    }
                    A_next[i * n + j] = sum;
                }
            }
            A = std::move(A_next);

            if (vecs) {
                std::vector<T> V_next(n * n, static_cast<T>(0));
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t j = 0; j < n; ++j) {
                        T sum = static_cast<T>(0);
                        for (int64_t k = 0; k < n; ++k) {
                            sum += V[i * n + k] * Q[k * n + j];
                        }
                        V_next[i * n + j] = sum;
                    }
                }
                V = std::move(V_next);
            }
        }

        // Extract eigenvalues from diagonal
        for (int64_t i = 0; i < n; ++i) {
            vals_r[i] = A[i * n + i];
            vals_i[i] = static_cast<T>(0);
        }

        if (vecs) {
            for (int64_t i = 0; i < n * n; ++i) {
                vecs[i] = V[i];
            }
        }
    }
}

} // namespace detail

std::pair<Tensor, Tensor> eig(const Tensor& input, bool eigenvectors) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::eig is currently supported only on CPU.");
    }

    const auto& shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) {
        throw std::invalid_argument("pygo::eig requires a tensor with at least 2 dimensions.");
    }

    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) {
        throw std::invalid_argument("pygo::eig requires the last two dimensions to be equal (square matrices).");
    }

    std::vector<int64_t> batch_shape(shape.begin(), shape.end() - 2);
    int64_t batch_size = 1;
    for (int64_t d : batch_shape) {
        batch_size *= d;
    }

    std::vector<int64_t> val_shape = batch_shape;
    val_shape.push_back(n);

    auto eigvals_real = zeros(Shape(val_shape), input.dtype(), input.device());
    auto eigvals_imag = zeros(Shape(val_shape), input.dtype(), input.device());
    
    Tensor eigvecs;
    if (eigenvectors) {
        eigvecs = zeros(Shape(shape), input.dtype(), input.device());
    } else {
        eigvecs = zeros({0}, input.dtype(), input.device());
    }

    if (input.dtype() == DType::Float32) {
        detail::eig_cpu_kernel<float>(
            input.data<float>(),
            eigvals_real.data<float>(),
            eigvals_imag.data<float>(),
            eigenvectors ? eigvecs.data<float>() : nullptr,
            batch_size, n
        );
    } else if (input.dtype() == DType::Float64) {
        detail::eig_cpu_kernel<double>(
            input.data<double>(),
            eigvals_real.data<double>(),
            eigvals_imag.data<double>(),
            eigenvectors ? eigvecs.data<double>() : nullptr,
            batch_size, n
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::eig. Expected floating-point type.");
    }

    return {eigvals_real, eigvecs};
}

} // namespace pygo
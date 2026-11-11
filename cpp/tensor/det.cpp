#include "pygo/det.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <utility>

namespace pygo {

namespace detail {

template <typename T>
void det_cpu_kernel(const T* src, T* dst, int64_t batch_size, int64_t n) {
    if (n == 0) {
        // Determinant of an empty matrix is 1.
        #pragma omp parallel for schedule(static)
        for (int64_t b = 0; b < batch_size; ++b) {
            dst[b] = static_cast<T>(1);
        }
        return;
    }

    #pragma omp parallel for schedule(guided)
    for (int64_t b = 0; b < batch_size; ++b) {
        // Create a local copy of the matrix for Gaussian elimination
        std::vector<T> mat(n * n);
        const T* src_matrix = src + b * n * n;
        for (int64_t i = 0; i < n * n; ++i) {
            mat[i] = src_matrix[i];
        }

        T det_val = static_cast<T>(1);
        bool singular = false;

        for (int64_t i = 0; i < n; ++i) {
            // Partial pivoting
            int64_t pivot_idx = i;
            T max_val = std::abs(mat[i * n + i]);
            for (int64_t j = i + 1; j < n; ++j) {
                T val = std::abs(mat[j * n + i]);
                if (val > max_val) {
                    max_val = val;
                    pivot_idx = j;
                }
            }

            if (max_val == static_cast<T>(0)) {
                singular = true;
                break;
            }

            if (pivot_idx != i) {
                // Swap rows i and pivot_idx
                for (int64_t c = i; c < n; ++c) {
                    std::swap(mat[i * n + c], mat[pivot_idx * n + c]);
                }
                det_val = -det_val;
            }

            T pivot_val = mat[i * n + i];
            det_val *= pivot_val;

            // Eliminate column i for rows below
            for (int64_t j = i + 1; j < n; ++j) {
                T factor = mat[j * n + i] / pivot_val;
                for (int64_t c = i + 1; c < n; ++c) {
                    mat[j * n + c] -= factor * mat[i * n + c];
                }
            }
        }

        dst[b] = singular ? static_cast<T>(0) : det_val;
    }
}

} // namespace detail

Tensor det(const Tensor& input) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::det is currently supported only on CPU.");
    }

    const auto& shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    if (ndim < 2) {
        throw std::invalid_argument("pygo::det requires a tensor with at least 2 dimensions.");
    }

    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) {
        throw std::invalid_argument("pygo::det requires the last two dimensions to be equal (square matrices).");
    }

    std::vector<int64_t> out_shape(shape.begin(), shape.end() - 2);
    int64_t batch_size = 1;
    for (int64_t d : out_shape) {
        batch_size *= d;
    }

    // Output is generally a batched tensor with the last two dimensions reduced. 
    // If input is 2D (M, M), output shape is empty (0D tensor).
    auto output = zeros(Shape(out_shape), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        detail::det_cpu_kernel<float>(
            input.data<float>(), output.data<float>(), batch_size, n
        );
    } else if (input.dtype() == DType::Float64) {
        detail::det_cpu_kernel<double>(
            input.data<double>(), output.data<double>(), batch_size, n
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::det. Expected floating-point type.");
    }

    return output;
}

} // namespace pygo
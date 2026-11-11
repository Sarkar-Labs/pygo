#include "pygo/fft.h"
#include <cmath>
#include <stdexcept>
#include <vector>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

template <typename T>
void fft_cpu_kernel(const T* src_real, T* dst_real, T* dst_imag, 
                    int64_t outer_size, int64_t dim_size, int64_t inner_size) {
    // Check if dim_size is a power of 2 for radix-2 Cooley-Tukey
    bool is_pow2 = (dim_size > 0) && ((dim_size & (dim_size - 1)) == 0);
    if (!is_pow2) {
        throw std::invalid_argument("pygo::fft currently requires dimension size to be a power of 2.");
    }

    int64_t levels = 0;
    int64_t temp = dim_size;
    while (temp > 1) {
        levels++;
        temp >>= 1;
    }

    #pragma omp parallel for collapse(2) schedule(static) if(outer_size * inner_size > 1024)
    for (int64_t o = 0; o < outer_size; ++o) {
        for (int64_t i = 0; i < inner_size; ++i) {
            // Allocate local buffers for the 1D signal along the transform axis
            std::vector<T> real_buf(dim_size);
            std::vector<T> imag_buf(dim_size, static_cast<T>(0));

            // Gather input data
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t src_idx = o * dim_size * inner_size + d * inner_size + i;
                real_buf[d] = src_real[src_idx];
            }

            // Bit-reversal permutation
            for (int64_t j = 0; j < dim_size; ++j) {
                int64_t k = 0;
                for (int64_t l = 0; l < levels; ++l) {
                    if ((j >> l) & 1) {
                        k |= (1 << (levels - 1 - l));
                    }
                }
                if (j < k) {
                    std::swap(real_buf[j], real_buf[k]);
                    std::swap(imag_buf[j], imag_buf[k]);
                }
            }

            // Cooley-Tukey computation
            for (int64_t l = 1; l <= levels; ++l) {
                int64_t m = 1 << l;
                int64_t m2 = m >> 1;
                T theta = static_cast<T>(-2.0 * M_PI / m);
                
                for (int64_t k = 0; k < dim_size; k += m) {
                    for (int64_t j = 0; j < m2; ++j) {
                        T w_real = std::cos(theta * j);
                        T w_imag = std::sin(theta * j);
                        
                        int64_t idx1 = k + j;
                        int64_t idx2 = k + j + m2;
                        
                        T u_r = real_buf[idx1];
                        T u_i = imag_buf[idx1];
                        
                        T v_r = real_buf[idx2] * w_real - imag_buf[idx2] * w_imag;
                        T v_i = real_buf[idx2] * w_imag + imag_buf[idx2] * w_real;
                        
                        real_buf[idx1] = u_r + v_r;
                        imag_buf[idx1] = u_i + v_i;
                        real_buf[idx2] = u_r - v_r;
                        imag_buf[idx2] = u_i - v_i;
                    }
                }
            }

            // Scatter output data
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t dst_idx = o * dim_size * inner_size + d * inner_size + i;
                dst_real[dst_idx] = real_buf[d];
                dst_imag[dst_idx] = imag_buf[d];
            }
        }
    }
}

} // namespace detail

std::pair<Tensor, Tensor> fft(const Tensor& input, int64_t dim) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::fft is currently supported only on CPU.");
    }

    int64_t ndim = static_cast<int64_t>(input.shape().ndim());
    if (ndim == 0) {
        throw std::invalid_argument("pygo::fft cannot be performed on a 0-dimensional tensor.");
    }

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("Dimension out of range in pygo::fft.");
    }

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= input.shape()[i];
    }

    int64_t dim_size = input.shape()[dim];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= input.shape()[i];
    }

    auto out_real = zeros(input.shape(), input.dtype(), input.device());
    auto out_imag = zeros(input.shape(), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        detail::fft_cpu_kernel<float>(
            input.data<float>(),
            out_real.data<float>(),
            out_imag.data<float>(),
            outer_size, dim_size, inner_size
        );
    } else if (input.dtype() == DType::Float64) {
        detail::fft_cpu_kernel<double>(
            input.data<double>(),
            out_real.data<double>(),
            out_imag.data<double>(),
            outer_size, dim_size, inner_size
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::fft. Expected floating-point type.");
    }

    return {out_real, out_imag};
}

} // namespace pygo
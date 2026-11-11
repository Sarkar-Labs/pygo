#include "pygo/ifft.h"
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
void ifft_cpu_kernel(const T* src_real, const T* src_imag, T* dst_real, T* dst_imag, 
                     int64_t outer_size, int64_t dim_size, int64_t inner_size) {
    bool is_pow2 = (dim_size > 0) && ((dim_size & (dim_size - 1)) == 0);
    if (!is_pow2) {
        throw std::invalid_argument("pygo::ifft currently requires dimension size to be a power of 2.");
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
            std::vector<T> real_buf(dim_size);
            std::vector<T> imag_buf(dim_size);

            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t src_idx = o * dim_size * inner_size + d * inner_size + i;
                real_buf[d] = src_real[src_idx];
                imag_buf[d] = src_imag[src_idx];
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

            // Cooley-Tukey computation with positive exponent for IFFT
            for (int64_t l = 1; l <= levels; ++l) {
                int64_t m = 1 << l;
                int64_t m2 = m >> 1;
                T theta = static_cast<T>(2.0 * M_PI / m);
                
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

            // Scale by 1 / dim_size and scatter output data
            T inv_n = static_cast<T>(1.0) / static_cast<T>(dim_size);
            for (int64_t d = 0; d < dim_size; ++d) {
                int64_t dst_idx = o * dim_size * inner_size + d * inner_size + i;
                dst_real[dst_idx] = real_buf[d] * inv_n;
                dst_imag[dst_idx] = imag_buf[d] * inv_n;
            }
        }
    }
}

} // namespace detail

std::pair<Tensor, Tensor> ifft(const Tensor& real, const Tensor& imag, int64_t dim) {
    if (!real.device().is_cpu() || !imag.device().is_cpu()) {
        throw std::runtime_error("pygo::ifft is currently supported only on CPU.");
    }

    if (real.dtype() != imag.dtype()) {
        throw std::invalid_argument("pygo::ifft expects real and imaginary tensors to have the same dtype.");
    }

    if (real.shape() != imag.shape()) {
        throw std::invalid_argument("pygo::ifft expects real and imaginary tensors to have the same shape.");
    }

    int64_t ndim = static_cast<int64_t>(real.shape().ndim());
    if (ndim == 0) {
        throw std::invalid_argument("pygo::ifft cannot be performed on a 0-dimensional tensor.");
    }

    if (dim < 0) dim += ndim;
    if (dim < 0 || dim >= ndim) {
        throw std::out_of_range("Dimension out of range in pygo::ifft.");
    }

    int64_t outer_size = 1;
    for (int64_t i = 0; i < dim; ++i) {
        outer_size *= real.shape()[i];
    }

    int64_t dim_size = real.shape()[dim];

    int64_t inner_size = 1;
    for (int64_t i = dim + 1; i < ndim; ++i) {
        inner_size *= real.shape()[i];
    }

    auto out_real = zeros(real.shape(), real.dtype(), real.device());
    auto out_imag = zeros(real.shape(), real.dtype(), real.device());

    if (real.dtype() == DType::Float32) {
        detail::ifft_cpu_kernel<float>(
            real.data<float>(),
            imag.data<float>(),
            out_real.data<float>(),
            out_imag.data<float>(),
            outer_size, dim_size, inner_size
        );
    } else if (real.dtype() == DType::Float64) {
        detail::ifft_cpu_kernel<double>(
            real.data<double>(),
            imag.data<double>(),
            out_real.data<double>(),
            out_imag.data<double>(),
            outer_size, dim_size, inner_size
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::ifft. Expected floating-point type.");
    }

    return {out_real, out_imag};
}

} // namespace pygo
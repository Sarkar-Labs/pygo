#include "pygo/conv1d.h"
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
void conv1d_cpu_kernel(const T* input,
                       const T* weight,
                       const T* bias,
                       T* output,
                       int64_t N, int64_t C_in, int64_t L_in,
                       int64_t C_out, int64_t K,
                       int64_t L_out,
                       int64_t stride, int64_t padding,
                       int64_t dilation, int64_t groups) {
    int64_t in_channels_per_group = C_in / groups;
    int64_t out_channels_per_group = C_out / groups;

    #pragma omp parallel for collapse(2) schedule(static) if(N * C_out > 64)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < groups; ++g) {
            int64_t out_ch_start = g * out_channels_per_group;
            int64_t in_ch_start = g * in_channels_per_group;

            for (int64_t oc_idx = 0; oc_idx < out_channels_per_group; ++oc_idx) {
                int64_t oc = out_ch_start + oc_idx;
                T b_val = bias ? bias[oc] : static_cast<T>(0);

                for (int64_t l = 0; l < L_out; ++l) {
                    T sum = b_val;
                    int64_t in_l_origin = l * stride - padding;

                    for (int64_t ic_idx = 0; ic_idx < in_channels_per_group; ++ic_idx) {
                        int64_t ic = in_ch_start + ic_idx;
                        const T* in_ptr = input + (n * C_in + ic) * L_in;
                        const T* w_ptr = weight + (oc * in_channels_per_group + ic_idx) * K;

                        for (int64_t k = 0; k < K; ++k) {
                            int64_t in_l = in_l_origin + k * dilation;
                            if (in_l >= 0 && in_l < L_in) {
                                sum += in_ptr[in_l] * w_ptr[k];
                            }
                        }
                    }

                    output[(n * C_out + oc) * L_out + l] = sum;
                }
            }
        }
    }
}

} // namespace detail

Tensor conv1d(const Tensor& input,
              const Tensor& weight,
              const std::optional<Tensor>& bias,
              int64_t stride,
              int64_t padding,
              int64_t dilation,
              int64_t groups) {
    if (!input.device().is_cpu() || !weight.device().is_cpu()) {
        throw std::runtime_error("pygo::conv1d is currently supported only on CPU.");
    }

    if (bias.has_value() && !bias->device().is_cpu()) {
        throw std::runtime_error("pygo::conv1d bias device mismatch.");
    }

    const auto& in_shape = input.shape();
    const auto& w_shape = weight.shape();

    if (in_shape.size() != 3) {
        throw std::invalid_argument("pygo::conv1d input must be a 3D tensor (N, C_in, L_in).");
    }
    if (w_shape.size() != 3) {
        throw std::invalid_argument("pygo::conv1d weight must be a 3D tensor (C_out, C_in/groups, K).");
    }

    int64_t N = in_shape[0];
    int64_t C_in = in_shape[1];
    int64_t L_in = in_shape[2];

    int64_t C_out = w_shape[0];
    int64_t in_ch_per_group = w_shape[1];
    int64_t K = w_shape[2];

    if (groups <= 0) {
        throw std::invalid_argument("pygo::conv1d groups must be positive.");
    }

    if (C_in % groups != 0 || C_out % groups != 0) {
        throw std::invalid_argument("pygo::conv1d C_in and C_out must be divisible by groups.");
    }

    if (in_ch_per_group != C_in / groups) {
        throw std::invalid_argument("pygo::conv1d weight shape does not match input channel count and groups.");
    }

    if (bias.has_value()) {
        const auto& b_shape = bias->shape();
        if (b_shape.size() != 1 || b_shape[0] != C_out) {
            throw std::invalid_argument("pygo::conv1d bias shape must be 1D with size C_out.");
        }
    }

    int64_t L_out = (L_in + 2 * padding - dilation * (K - 1) - 1) / stride + 1;
    if (L_out <= 0) {
        throw std::invalid_argument("pygo::conv1d output length is less than or equal to zero.");
    }

    auto output = zeros({N, C_out, L_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* b_ptr = bias.has_value() ? bias->data<float>() : nullptr;
        detail::conv1d_cpu_kernel<float>(
            input.data<float>(), weight.data<float>(), b_ptr, output.data<float>(),
            N, C_in, L_in, C_out, K, L_out, stride, padding, dilation, groups
        );
    } else if (input.dtype() == DType::Float64) {
        const double* b_ptr = bias.has_value() ? bias->data<double>() : nullptr;
        detail::conv1d_cpu_kernel<double>(
            input.data<double>(), weight.data<double>(), b_ptr, output.data<double>(),
            N, C_in, L_in, C_out, K, L_out, stride, padding, dilation, groups
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::conv1d.");
    }

    return output;
}

} // namespace pygo
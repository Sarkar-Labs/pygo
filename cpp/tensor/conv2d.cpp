#include "pygo/conv2d.h"
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
void conv2d_cpu_kernel(const T* input,
                       const T* weight,
                       const T* bias,
                       T* output,
                       int64_t N, int64_t C_in, int64_t H_in, int64_t W_in,
                       int64_t C_out, int64_t KH, int64_t KW,
                       int64_t H_out, int64_t W_out,
                       int64_t stride_h, int64_t stride_w,
                       int64_t pad_h, int64_t pad_w,
                       int64_t dilation_h, int64_t dilation_w,
                       int64_t groups) {
    int64_t in_channels_per_group = C_in / groups;
    int64_t out_channels_per_group = C_out / groups;

    #pragma omp parallel for collapse(2) schedule(static) if(N * C_out > 32)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < groups; ++g) {
            int64_t out_ch_start = g * out_channels_per_group;
            int64_t in_ch_start = g * in_channels_per_group;

            for (int64_t oc_idx = 0; oc_idx < out_channels_per_group; ++oc_idx) {
                int64_t oc = out_ch_start + oc_idx;
                T b_val = bias ? bias[oc] : static_cast<T>(0);

                for (int64_t oh = 0; oh < H_out; ++oh) {
                    int64_t in_h_origin = oh * stride_h - pad_h;

                    for (int64_t ow = 0; ow < W_out; ++ow) {
                        T sum = b_val;
                        int64_t in_w_origin = ow * stride_w - pad_w;

                        for (int64_t ic_idx = 0; ic_idx < in_channels_per_group; ++ic_idx) {
                            int64_t ic = in_ch_start + ic_idx;
                            const T* in_ptr = input + ((n * C_in + ic) * H_in) * W_in;
                            const T* w_ptr = weight + (((oc * in_channels_per_group + ic_idx) * KH)) * KW;

                            for (int64_t kh = 0; kh < KH; ++kh) {
                                int64_t ih = in_h_origin + kh * dilation_h;
                                if (ih >= 0 && ih < H_in) {
                                    for (int64_t kw = 0; kw < KW; ++kw) {
                                        int64_t iw = in_w_origin + kw * dilation_w;
                                        if (iw >= 0 && iw < W_in) {
                                            sum += in_ptr[ih * W_in + iw] * w_ptr[kh * KW + kw];
                                        }
                                    }
                                }
                            }
                        }

                        output[((n * C_out + oc) * H_out + oh) * W_out + ow] = sum;
                    }
                }
            }
        }
    }
}

} // namespace detail

Tensor conv2d(const Tensor& input,
              const Tensor& weight,
              const std::optional<Tensor>& bias,
              std::pair<int64_t, int64_t> stride,
              std::pair<int64_t, int64_t> padding,
              std::pair<int64_t, int64_t> dilation,
              int64_t groups) {
    if (!input.device().is_cpu() || !weight.device().is_cpu()) {
        throw std::runtime_error("pygo::conv2d is currently supported only on CPU.");
    }

    if (bias.has_value() && !bias->device().is_cpu()) {
        throw std::runtime_error("pygo::conv2d bias device mismatch.");
    }

    const auto& in_shape = input.shape();
    const auto& w_shape = weight.shape();

    if (in_shape.size() != 4) {
        throw std::invalid_argument("pygo::conv2d input must be a 4D tensor (N, C_in, H_in, W_in).");
    }
    if (w_shape.size() != 4) {
        throw std::invalid_argument("pygo::conv2d weight must be a 4D tensor (C_out, C_in/groups, KH, KW).");
    }

    int64_t N = in_shape[0];
    int64_t C_in = in_shape[1];
    int64_t H_in = in_shape[2];
    int64_t W_in = in_shape[3];

    int64_t C_out = w_shape[0];
    int64_t in_ch_per_group = w_shape[1];
    int64_t KH = w_shape[2];
    int64_t KW = w_shape[3];

    auto [stride_h, stride_w] = stride;
    auto [pad_h, pad_w] = padding;
    auto [dilation_h, dilation_w] = dilation;

    if (groups <= 0) {
        throw std::invalid_argument("pygo::conv2d groups must be positive.");
    }

    if (C_in % groups != 0 || C_out % groups != 0) {
        throw std::invalid_argument("pygo::conv2d C_in and C_out must be divisible by groups.");
    }

    if (in_ch_per_group != C_in / groups) {
        throw std::invalid_argument("pygo::conv2d weight shape does not match input channel count and groups.");
    }

    if (bias.has_value()) {
        const auto& b_shape = bias->shape();
        if (b_shape.size() != 1 || b_shape[0] != C_out) {
            throw std::invalid_argument("pygo::conv2d bias shape must be 1D with size C_out.");
        }
    }

    int64_t H_out = (H_in + 2 * pad_h - dilation_h * (KH - 1) - 1) / stride_h + 1;
    int64_t W_out = (W_in + 2 * pad_w - dilation_w * (KW - 1) - 1) / stride_w + 1;

    if (H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("pygo::conv2d output spatial dimensions must be positive.");
    }

    auto output = zeros({N, C_out, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* b_ptr = bias.has_value() ? bias->data<float>() : nullptr;
        detail::conv2d_cpu_kernel<float>(
            input.data<float>(), weight.data<float>(), b_ptr, output.data<float>(),
            N, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups
        );
    } else if (input.dtype() == DType::Float64) {
        const double* b_ptr = bias.has_value() ? bias->data<double>() : nullptr;
        detail::conv2d_cpu_kernel<double>(
            input.data<double>(), weight.data<double>(), b_ptr, output.data<double>(),
            N, C_in, H_in, W_in, C_out, KH, KW, H_out, W_out,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w, groups
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::conv2d.");
    }

    return output;
}

} // namespace pygo
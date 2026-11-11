#include "pygo/conv3d.h"
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
void conv3d_cpu_kernel(const T* input,
                       const T* weight,
                       const T* bias,
                       T* output,
                       int64_t N, int64_t C_in, int64_t D_in, int64_t H_in, int64_t W_in,
                       int64_t C_out, int64_t KD, int64_t KH, int64_t KW,
                       int64_t D_out, int64_t H_out, int64_t W_out,
                       int64_t stride_d, int64_t stride_h, int64_t stride_w,
                       int64_t pad_d, int64_t pad_h, int64_t pad_w,
                       int64_t dilation_d, int64_t dilation_h, int64_t dilation_w,
                       int64_t groups) {
    int64_t in_channels_per_group = C_in / groups;
    int64_t out_channels_per_group = C_out / groups;

    #pragma omp parallel for collapse(2) schedule(static) if(N * C_out > 16)
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t g = 0; g < groups; ++g) {
            int64_t out_ch_start = g * out_channels_per_group;
            int64_t in_ch_start = g * in_channels_per_group;

            for (int64_t oc_idx = 0; oc_idx < out_channels_per_group; ++oc_idx) {
                int64_t oc = out_ch_start + oc_idx;
                T b_val = bias ? bias[oc] : static_cast<T>(0);

                for (int64_t od = 0; od < D_out; ++od) {
                    int64_t in_d_origin = od * stride_d - pad_d;

                    for (int64_t oh = 0; oh < H_out; ++oh) {
                        int64_t in_h_origin = oh * stride_h - pad_h;

                        for (int64_t ow = 0; ow < W_out; ++ow) {
                            T sum = b_val;
                            int64_t in_w_origin = ow * stride_w - pad_w;

                            for (int64_t ic_idx = 0; ic_idx < in_channels_per_group; ++ic_idx) {
                                int64_t ic = in_ch_start + ic_idx;
                                const T* in_ptr = input + (((n * C_in + ic) * D_in) * H_in) * W_in;
                                const T* w_ptr = weight + ((((oc * in_channels_per_group + ic_idx) * KD)) * KH) * KW;

                                for (int64_t kd = 0; kd < KD; ++kd) {
                                    int64_t id = in_d_origin + kd * dilation_d;
                                    if (id >= 0 && id < D_in) {
                                        for (int64_t kh = 0; kh < KH; ++kh) {
                                            int64_t ih = in_h_origin + kh * dilation_h;
                                            if (ih >= 0 && ih < H_in) {
                                                for (int64_t kw = 0; kw < KW; ++kw) {
                                                    int64_t iw = in_w_origin + kw * dilation_w;
                                                    if (iw >= 0 && iw < W_in) {
                                                        sum += in_ptr[(id * H_in + ih) * W_in + iw] * w_ptr[(kd * KH + kh) * KW + kw];
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            output[((((n * C_out + oc) * D_out) + od) * H_out + oh) * W_out + ow] = sum;
                        }
                    }
                }
            }
        }
    }
}

} // namespace detail

Tensor conv3d(const Tensor& input,
              const Tensor& weight,
              const std::optional<Tensor>& bias,
              std::tuple<int64_t, int64_t, int64_t> stride,
              std::tuple<int64_t, int64_t, int64_t> padding,
              std::tuple<int64_t, int64_t, int64_t> dilation,
              int64_t groups) {
    if (!input.device().is_cpu() || !weight.device().is_cpu()) {
        throw std::runtime_error("pygo::conv3d is currently supported only on CPU.");
    }

    if (bias.has_value() && !bias->device().is_cpu()) {
        throw std::runtime_error("pygo::conv3d bias device mismatch.");
    }

    const auto& in_shape = input.shape();
    const auto& w_shape = weight.shape();

    if (in_shape.size() != 5) {
        throw std::invalid_argument("pygo::conv3d input must be a 5D tensor (N, C_in, D_in, H_in, W_in).");
    }
    if (w_shape.size() != 5) {
        throw std::invalid_argument("pygo::conv3d weight must be a 5D tensor (C_out, C_in/groups, KD, KH, KW).");
    }

    int64_t N = in_shape[0];
    int64_t C_in = in_shape[1];
    int64_t D_in = in_shape[2];
    int64_t H_in = in_shape[3];
    int64_t W_in = in_shape[4];

    int64_t C_out = w_shape[0];
    int64_t in_ch_per_group = w_shape[1];
    int64_t KD = w_shape[2];
    int64_t KH = w_shape[3];
    int64_t KW = w_shape[4];

    auto [stride_d, stride_h, stride_w] = stride;
    auto [pad_d, pad_h, pad_w] = padding;
    auto [dilation_d, dilation_h, dilation_w] = dilation;

    if (groups <= 0) {
        throw std::invalid_argument("pygo::conv3d groups must be positive.");
    }

    if (C_in % groups != 0 || C_out % groups != 0) {
        throw std::invalid_argument("pygo::conv3d C_in and C_out must be divisible by groups.");
    }

    if (in_ch_per_group != C_in / groups) {
        throw std::invalid_argument("pygo::conv3d weight shape does not match input channel count and groups.");
    }

    if (bias.has_value()) {
        const auto& b_shape = bias->shape();
        if (b_shape.size() != 1 || b_shape[0] != C_out) {
            throw std::invalid_argument("pygo::conv3d bias shape must be 1D with size C_out.");
        }
    }

    int64_t D_out = (D_in + 2 * pad_d - dilation_d * (KD - 1) - 1) / stride_d + 1;
    int64_t H_out = (H_in + 2 * pad_h - dilation_h * (KH - 1) - 1) / stride_h + 1;
    int64_t W_out = (W_in + 2 * pad_w - dilation_w * (KW - 1) - 1) / stride_w + 1;

    if (D_out <= 0 || H_out <= 0 || W_out <= 0) {
        throw std::invalid_argument("pygo::conv3d output spatial dimensions must be positive.");
    }

    auto output = zeros({N, C_out, D_out, H_out, W_out}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* b_ptr = bias.has_value() ? bias->data<float>() : nullptr;
        detail::conv3d_cpu_kernel<float>(
            input.data<float>(), weight.data<float>(), b_ptr, output.data<float>(),
            N, C_in, D_in, H_in, W_in, C_out, KD, KH, KW, D_out, H_out, W_out,
            stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w, groups
        );
    } else if (input.dtype() == DType::Float64) {
        const double* b_ptr = bias.has_value() ? bias->data<double>() : nullptr;
        detail::conv3d_cpu_kernel<double>(
            input.data<double>(), weight.data<double>(), b_ptr, output.data<double>(),
            N, C_in, D_in, H_in, W_in, C_out, KD, KH, KW, D_out, H_out, W_out,
            stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, dilation_d, dilation_h, dilation_w, groups
        );
    } else {
        throw std::runtime_error("Unsupported DType for pygo::conv3d.");
    }

    return output;
}

} // namespace pygo
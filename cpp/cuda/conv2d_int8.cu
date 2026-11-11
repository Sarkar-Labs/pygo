#include "pygo/cuda.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

namespace pygo::cuda {

namespace {

__global__ void conv2d_int8_kernel(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weight,
    const int8_t* __restrict__ bias,
    int8_t* __restrict__ output,
    int N, int C_in, int H, int W,
    int C_out, int kH, int kW,
    int H_out, int W_out,
    int stride_h, int stride_w,
    int pad_h, int pad_w,
    int dilation_h, int dilation_w,
    bool has_bias) {

    int index = blockDim.x * blockIdx.x + threadIdx.x;
    int total_elements = N * C_out * H_out * W_out;
    if (index >= total_elements) return;

    int w_out = index % W_out;
    int h_out = (index / W_out) % H_out;
    int c_out = (index / (W_out * H_out)) % C_out;
    int n     = index / (W_out * H_out * C_out);

    int acc = 0;
    if (has_bias && bias != nullptr) {
        acc = static_cast<int>(bias[c_out]);
    }

    int in_batch_offset = n * C_in * H * W;
    int weight_out_offset = c_out * C_in * kH * kW;

    for (int c_in = 0; c_in < C_in; ++c_in) {
        int in_channel_offset = in_batch_offset + c_in * H * W;
        int weight_channel_offset = weight_out_offset + c_in * kH * kW;

        for (int kh = 0; kh < kH; ++kh) {
            int h_in = h_out * stride_h - pad_h + kh * dilation_h;
            if (h_in < 0 || h_in >= H) continue;

            for (int kw = 0; kw < kW; ++kw) {
                int w_in = w_out * stride_w - pad_w + kw * dilation_w;
                if (w_in < 0 || w_in >= W) continue;

                int in_val = static_cast<int>(input[in_channel_offset + h_in * W + w_in]);
                int wt_val = static_cast<int>(weight[weight_channel_offset + kh * kW + kw]);
                acc += in_val * wt_val;
            }
        }
    }

    // Simple clamping/saturation for INT8
    if (acc > 127) acc = 127;
    else if (acc < -128) acc = -128;

    output[index] = static_cast<int8_t>(acc);
}

} // anonymous namespace

Tensor conv2d_int8(const Tensor& input, const Tensor& weight, const Tensor& bias,
                   int stride_h, int stride_w, int pad_h, int pad_w,
                   int dilation_h, int dilation_w) {
    if (input.ndim() != 4 || weight.ndim() != 4) {
        throw std::invalid_argument("pygo::cuda::conv2d_int8: input and weight must be 4D tensors");
    }

    int64_t N = input.shape().dims[0];
    int64_t C_in = input.shape().dims[1];
    int64_t H = input.shape().dims[2];
    int64_t W = input.shape().dims[3];

    int64_t C_out = weight.shape().dims[0];
    int64_t w_c_in = weight.shape().dims[1];
    int64_t kH = weight.shape().dims[2];
    int64_t kW = weight.shape().dims[3];

    if (C_in != w_c_in) {
        throw std::invalid_argument("pygo::cuda::conv2d_int8: input channel mismatch between input and weight");
    }

    bool has_bias = bias.numel() > 0;
    if (has_bias && (bias.ndim() != 1 || bias.shape().dims[0] != C_out)) {
        throw std::invalid_argument("pygo::cuda::conv2d_int8: bias must be 1D with size equal to C_out");
    }

    int64_t H_out = (H + 2 * pad_h - dilation_h * (kH - 1) - 1) / stride_h + 1;
    int64_t W_out = (W + 2 * pad_w - dilation_w * (kW - 1) - 1) / stride_w + 1;

    if (H_out <= 0 || W_out <= 0) {
        throw std::runtime_error("pygo::cuda::conv2d_int8: invalid output dimensions");
    }

    if (!is_available())
        throw std::runtime_error("pygo::cuda::conv2d_int8: no CUDA device available");

    Tensor input_c = input.contiguous();
    Tensor weight_c = weight.contiguous();
    Tensor bias_c = has_bias ? bias.contiguous() : Tensor();

    size_t input_bytes = N * C_in * H * W * sizeof(int8_t);
    size_t weight_bytes = C_out * C_in * kH * kW * sizeof(int8_t);
    size_t bias_bytes = has_bias ? C_out * sizeof(int8_t) : 0;
    size_t output_bytes = N * C_out * H_out * W_out * sizeof(int8_t);

    int8_t* d_input = static_cast<int8_t*>(to_device(input_c.data<void>(), input_bytes));
    int8_t* d_weight = static_cast<int8_t*>(to_device(weight_c.data<void>(), weight_bytes));
    int8_t* d_bias = has_bias ? static_cast<int8_t*>(to_device(bias_c.data<void>(), bias_bytes)) : nullptr;
    int8_t* d_output = nullptr;

    try {
        PYGO_CUDA_CHECK(cudaMalloc(&d_output, output_bytes));

        int total_elements = (int)(N * C_out * H_out * W_out);
        int threads = 256;
        int blocks = (total_elements + threads - 1) / threads;

        conv2d_int8_kernel<<<blocks, threads>>>(
            d_input, d_weight, d_bias, d_output,
            (int)N, (int)C_in, (int)H, (int)W,
            (int)C_out, (int)kH, (int)kW,
            (int)H_out, (int)W_out,
            stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w,
            has_bias
        );

        PYGO_CUDA_CHECK(cudaGetLastError());
        PYGO_CUDA_CHECK(cudaDeviceSynchronize());

        std::vector<int64_t> out_shape = {N, C_out, H_out, W_out};
        Tensor out = empty(Shape(out_shape), DType::Int8, Device::CPU);
        to_host(out.mutable_data<void>(), d_output, output_bytes);

        free_device(d_input);
        free_device(d_weight);
        if (d_bias) free_device(d_bias);
        free_device(d_output);

        return out;
    } catch (...) {
        free_device(d_input);
        free_device(d_weight);
        if (d_bias) free_device(d_bias);
        if (d_output) free_device(d_output);
        throw;
    }
}

} // namespace pygo::cuda
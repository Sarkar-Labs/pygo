#include "pygo/cuda.h"
#include <cmath>
#include <stdexcept>

namespace pygo::cuda {

namespace {

__global__ void adamw_fp32_kernel(
    float* param,
    const float* grad,
    float* m,
    float* v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bc1_inv,
    float bc2_inv,
    int64_t numel) {

    int64_t idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= numel) return;

    float p = param[idx];
    float g = grad[idx];

    if (weight_decay != 0.0f) {
        g += weight_decay * p;
    }

    float m_val = beta1 * m[idx] + (1.0f - beta1) * g;
    float v_val = beta2 * v[idx] + (1.0f - beta2) * g * g;

    m[idx] = m_val;
    v[idx] = v_val;

    float m_hat = m_val * bc1_inv;
    float v_hat = v_val * bc2_inv;

    p -= lr * (m_hat / (sqrtf(v_hat) + eps));

    param[idx] = p;
}

} // anonymous namespace

void adamw_fp32(
    Tensor& param,
    const Tensor& grad,
    Tensor& m,
    Tensor& v,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    int64_t step) {

    if (!is_available())
        throw std::runtime_error("pygo::cuda::adamw_fp32: no CUDA device available");

    int64_t numel = param.numel();
    if (grad.numel() != numel || m.numel() != numel || v.numel() != numel)
        throw std::invalid_argument("pygo::cuda::adamw_fp32: tensor size mismatch");
    if (param.dtype() != DType::Float32 || grad.dtype() != DType::Float32 ||
        m.dtype() != DType::Float32 || v.dtype() != DType::Float32)
        throw std::invalid_argument("pygo::cuda::adamw_fp32: all tensors must be Float32");

    size_t f32_bytes = numel * sizeof(float);

    Tensor param_c = param.contiguous();
    Tensor grad_c = grad.contiguous();
    Tensor m_c = m.contiguous();
    Tensor v_c = v.contiguous();

    float* d_param = static_cast<float*>(to_device(param_c.data<float>(), f32_bytes));
    const float* d_grad = static_cast<const float*>(to_device(grad_c.data<float>(), f32_bytes));
    float* d_m = static_cast<float*>(to_device(m_c.data<float>(), f32_bytes));
    float* d_v = static_cast<float*>(to_device(v_c.data<float>(), f32_bytes));

    try {
        float bc1_inv = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(step)));
        float bc2_inv = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(step)));

        int threads = 256;
        int blocks = (int)((numel + threads - 1) / threads);

        adamw_fp32_kernel<<<blocks, threads>>>(
            d_param, d_grad, d_m, d_v,
            lr, beta1, beta2, eps, weight_decay,
            bc1_inv, bc2_inv, numel
        );

        PYGO_CUDA_CHECK(cudaGetLastError());
        PYGO_CUDA_CHECK(cudaDeviceSynchronize());

        to_host(param.mutable_data<float>(), d_param, f32_bytes);
        to_host(m.mutable_data<float>(), d_m, f32_bytes);
        to_host(v.mutable_data<float>(), d_v, f32_bytes);

        free_device(d_param);
        free_device(const_cast<float*>(d_grad));
        free_device(d_m);
        free_device(d_v);
    } catch (...) {
        free_device(d_param);
        free_device(const_cast<float*>(d_grad));
        free_device(d_m);
        free_device(d_v);
        throw;
    }
}

} // namespace pygo::cuda
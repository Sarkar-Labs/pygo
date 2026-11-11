#include "pygo/cuda.h"
#include <cuda_bf16.h>
#include <cmath>
#include <stdexcept>

namespace pygo::cuda {

namespace {

__global__ void adamw_bf16_kernel(
    __nv_bfloat16* param,
    const __nv_bfloat16* grad,
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

    float p = __bfloat162float(param[idx]);
    float g = __bfloat162float(grad[idx]);

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

    param[idx] = __float2bfloat16(p);
}

} // anonymous namespace

void adamw_bf16(
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
        throw std::runtime_error("pygo::cuda::adamw_bf16: no CUDA device available");

    int64_t numel = param.numel();
    if (grad.numel() != numel || m.numel() != numel || v.numel() != numel)
        throw std::invalid_argument("pygo::cuda::adamw_bf16: tensor size mismatch");

    size_t bf16_bytes = numel * sizeof(__nv_bfloat16);
    size_t f32_bytes = numel * sizeof(float);

    Tensor param_c = param.contiguous();
    Tensor grad_c = grad.contiguous();
    Tensor m_c = m.contiguous();
    Tensor v_c = v.contiguous();

    __nv_bfloat16* d_param = static_cast<__nv_bfloat16*>(to_device(param_c.data<void>(), bf16_bytes));
    const __nv_bfloat16* d_grad = static_cast<const __nv_bfloat16*>(to_device(grad_c.data<void>(), bf16_bytes));
    float* d_m = static_cast<float*>(to_device(m_c.data<float>(), f32_bytes));
    float* d_v = static_cast<float*>(to_device(v_c.data<float>(), f32_bytes));

    try {
        float bc1_inv = 1.0f / (1.0f - std::pow(beta1, static_cast<float>(step)));
        float bc2_inv = 1.0f / (1.0f - std::pow(beta2, static_cast<float>(step)));

        int threads = 256;
        int blocks = (int)((numel + threads - 1) / threads);

        adamw_bf16_kernel<<<blocks, threads>>>(
            d_param, d_grad, d_m, d_v,
            lr, beta1, beta2, eps, weight_decay,
            bc1_inv, bc2_inv, numel
        );

        PYGO_CUDA_CHECK(cudaGetLastError());
        PYGO_CUDA_CHECK(cudaDeviceSynchronize());

        to_host(param.mutable_data<void>(), d_param, bf16_bytes);
        to_host(m.mutable_data<float>(), d_m, f32_bytes);
        to_host(v.mutable_data<float>(), d_v, f32_bytes);

        free_device(d_param);
        free_device(const_cast<__nv_bfloat16*>(d_grad));
        free_device(d_m);
        free_device(d_v);
    } catch (...) {
        free_device(d_param);
        free_device(const_cast<__nv_bfloat16*>(d_grad));
        free_device(d_m);
        free_device(d_v);
        throw;
    }
}

} // namespace pygo::cuda
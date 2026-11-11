#pragma once
#include <cstddef>
#include <cstdint>
namespace pygo::kernels {
struct Status { int code; const char* message; explicit operator bool() const { return code == 0; } };
Status unary_exp(const float*, float*, std::size_t);
Status unary_log(const float*, float*, std::size_t);
Status unary_sin(const float*, float*, std::size_t);
Status unary_cos(const float*, float*, std::size_t);
Status unary_tanh(const float*, float*, std::size_t);
Status unary_sigmoid(const float*, float*, std::size_t);
Status unary_relu(const float*, float*, std::size_t);
Status unary_gelu(const float*, float*, std::size_t);
Status unary_silu(const float*, float*, std::size_t);
Status unary_mish(const float*, float*, std::size_t);
Status unary_softplus(const float*, float*, std::size_t);
Status unary_abs(const float*, float*, std::size_t);
Status unary_sqrt(const float*, float*, std::size_t);
Status unary_reciprocal(const float*, float*, std::size_t);
Status unary_square(const float*, float*, std::size_t);
Status unary_neg(const float*, float*, std::size_t);
Status binary_add(const float*, const float*, float*, std::size_t);
Status binary_sub(const float*, const float*, float*, std::size_t);
Status binary_mul(const float*, const float*, float*, std::size_t);
Status binary_div(const float*, const float*, float*, std::size_t);
Status binary_max(const float*, const float*, float*, std::size_t);
Status binary_min(const float*, const float*, float*, std::size_t);
Status reduce_sum(const float*, std::size_t, float*);
Status reduce_mean(const float*, std::size_t, float*);
Status reduce_max(const float*, std::size_t, float*);
Status reduce_min(const float*, std::size_t, float*);
Status reduce_l2(const float*, std::size_t, float*);
Status reduce_variance(const float*, std::size_t, float*);
Status matmul_f32(const float*, const float*, float*, std::size_t, std::size_t, std::size_t);
Status gemv_f32(const float*, const float*, float*, std::size_t, std::size_t);
Status transpose2d_f32(const float*, float*, std::size_t, std::size_t);
Status softmax_lastdim_f32(const float*, float*, std::size_t, std::size_t);
Status layernorm_lastdim_f32(const float*, const float*, const float*, float*, std::size_t, std::size_t, float);
}

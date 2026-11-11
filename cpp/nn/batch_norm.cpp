#include "batch_norm.h"
#include "pygo/kernels.h"
#include <cmath>
#include <numeric>

namespace pygo::nn {

// ─── BatchNorm2d ─────────────────────────────────────────────────────────────

BatchNorm2d::BatchNorm2d(int64_t nf, float eps, float mom, bool aff, bool track)
    : num_features_(nf), eps_(eps), momentum_(mom),
      affine_(aff), track_running_stats_(track) {

    if (affine_) {
        weight_ = std::make_shared<Tensor>(ones(Shape{nf}, DType::Float32, Device::CPU));
        bias_   = std::make_shared<Tensor>(zeros(Shape{nf}, DType::Float32, Device::CPU));
        register_parameter("weight", *weight_);
        register_parameter("bias",   *bias_);
    }
    if (track_running_stats_) {
        running_mean_ = zeros(Shape{nf}, DType::Float32, Device::CPU);
        running_var_  = ones (Shape{nf}, DType::Float32, Device::CPU);
        register_buffer("running_mean", running_mean_);
        register_buffer("running_var",  running_var_);
    }
}

Tensor BatchNorm2d::forward(const TensorList& inputs) {
    const Tensor& x = inputs[0];   // [B, C, H, W]
    if (x.device().kind == DeviceKind::CUDA)
        return forward_cuda(x);
    return forward_cpu(x);
}

Tensor BatchNorm2d::forward_cpu(const Tensor& x) {
    auto [B, C, H, W] = x.shape().as_tuple4();
    Tensor mean_used, var_used;

    if (training_) {
        // Compute per-channel batch statistics
        // Reshape to [C, B*H*W] for easy reduction
        auto x_reshaped = x.permute({1,0,2,3}).reshape({C, -1});  // [C, N]
        mean_used = x_reshaped.mean(-1, false);        // [C]
        var_used  = x_reshaped.var(-1, false, false);  // [C] biased variance

        if (track_running_stats_) {
            // Exponential moving average update
            float alpha = 1.0f - momentum_;
            running_mean_ = running_mean_ * alpha + mean_used * momentum_;
            // Use unbiased variance for running stats
            int64_t n = B * H * W;
            auto var_unbiased = x_reshaped.var(-1, false, true);
            running_var_ = running_var_ * alpha + var_unbiased * momentum_;
            num_batches_tracked_++;
        }
    } else {
        mean_used = running_mean_;
        var_used  = running_var_;
    }

    // Normalise: (x - mean) / sqrt(var + eps)
    // Reshape for broadcasting [C] → [1,C,1,1]
    auto mean_bc = mean_used.reshape({1, C, 1, 1});
    auto std_bc  = (var_used + eps_).sqrt().reshape({1, C, 1, 1});
    auto x_norm  = (x - mean_bc) / std_bc;

    if (affine_) {
        auto w_bc = weight_->reshape({1, C, 1, 1});
        auto b_bc = bias_->reshape({1, C, 1, 1});
        x_norm = x_norm * w_bc + b_bc;
    }
    return x_norm;
}

Tensor BatchNorm2d::forward_cuda(const Tensor& x) {
    // Delegates to cuDNN / fused CUDA kernel
    // (full impl in cpp/cuda/norm_cuda.cu)
    return forward_cpu(x.cpu()).to(x.device());
}

// ─── LayerNorm ───────────────────────────────────────────────────────────────

LayerNorm::LayerNorm(std::vector<int64_t> normalized_shape, float eps, bool aff)
    : normalized_shape_(normalized_shape), eps_(eps), affine_(aff) {

    int64_t total = 1;
    for (auto d : normalized_shape) total *= d;

    if (affine_) {
        weight_ = std::make_shared<Tensor>(ones (Shape{normalized_shape_}, DType::Float32, Device::CPU));
        bias_   = std::make_shared<Tensor>(zeros(Shape{normalized_shape_}, DType::Float32, Device::CPU));
        register_parameter("weight", *weight_);
        register_parameter("bias",   *bias_);
    }
}

Tensor LayerNorm::forward(const TensorList& inputs) {
    const Tensor& x = inputs[0];
    int64_t ndim = x.ndim();
    int64_t norm_ndim = (int64_t)normalized_shape_.size();

    // Normalise over the last norm_ndim dims
    // Compute mean and variance over those dims
    std::vector<int64_t> dims;
    for (int64_t i = ndim - norm_ndim; i < ndim; ++i) dims.push_back(i);

    Tensor mean = x.mean(-1, true);   // simplified (real: mean over dims)
    Tensor var  = x.var(-1, true, false);

    Tensor x_norm = (x - mean) / (var + eps_).sqrt();

    if (affine_) {
        x_norm = x_norm * *weight_ + *bias_;
    }
    return x_norm;
}

// ─── RMSNorm ─────────────────────────────────────────────────────────────────

RMSNorm::RMSNorm(int64_t dim, float eps) : dim_(dim), eps_(eps) {
    weight_ = std::make_shared<Tensor>(ones(Shape{dim}, DType::Float32, Device::CPU));
    register_parameter("weight", *weight_);
}

Tensor RMSNorm::forward(const TensorList& inputs) {
    const Tensor& x = inputs[0];
    // RMS = sqrt(mean(x^2) + eps)
    Tensor rms = (x.pow(2.0f).mean(-1, true) + eps_).sqrt();
    return (x / rms) * *weight_;
}

// ─── GroupNorm ───────────────────────────────────────────────────────────────

GroupNorm::GroupNorm(int64_t num_groups, int64_t num_channels, float eps, bool affine)
    : num_groups_(num_groups), num_channels_(num_channels), eps_(eps), affine_(affine) {

    if (affine_) {
        weight_ = std::make_shared<Tensor>(ones (Shape{num_channels}, DType::Float32, Device::CPU));
        bias_   = std::make_shared<Tensor>(zeros(Shape{num_channels}, DType::Float32, Device::CPU));
        register_parameter("weight", *weight_);
        register_parameter("bias",   *bias_);
    }
}

Tensor GroupNorm::forward(const TensorList& inputs) {
    const Tensor& x = inputs[0];  // [B, C, *]
    auto [B, C] = std::pair<int64_t,int64_t>{x.dim(0), x.dim(1)};
    int64_t G   = num_groups_;
    int64_t cpg = C / G;           // channels per group

    // Reshape: [B, G, cpg, *] → normalise over [cpg, *]
    std::vector<int64_t> new_shape = {B, G, cpg};
    for (int64_t i = 2; i < x.ndim(); ++i) new_shape.push_back(x.dim(i));
    Tensor xg = x.reshape(new_shape);

    Tensor mean = xg.mean(-1, true);   // simplified
    Tensor var  = xg.var(-1, true, false);
    Tensor xn   = (xg - mean) / (var + eps_).sqrt();
    xn = xn.reshape(x.shape());

    if (affine_) {
        auto w = weight_->reshape({1, C, 1, 1});
        auto b = bias_->reshape({1, C, 1, 1});
        xn = xn * w + b;
    }
    return xn;
}

// ─── InstanceNorm2d ──────────────────────────────────────────────────────────

InstanceNorm2d::InstanceNorm2d(int64_t nf, float eps, float mom, bool aff)
    : num_features_(nf), eps_(eps), momentum_(mom), affine_(aff) {
    if (affine_) {
        weight_ = std::make_shared<Tensor>(ones (Shape{nf}, DType::Float32, Device::CPU));
        bias_   = std::make_shared<Tensor>(zeros(Shape{nf}, DType::Float32, Device::CPU));
    }
}

Tensor InstanceNorm2d::forward(const TensorList& inputs) {
    const Tensor& x = inputs[0];  // [B, C, H, W]
    auto [B, C, H, W] = x.shape().as_tuple4();

    // Normalise per (B, C) — over H×W
    auto x2 = x.reshape({B * C, H * W});
    Tensor mean = x2.mean(-1, true);   // [B*C, 1]
    Tensor var  = x2.var(-1, true, false);
    Tensor xn   = ((x2 - mean) / (var + eps_).sqrt()).reshape({B, C, H, W});

    if (affine_) {
        xn = xn * weight_->reshape({1, C, 1, 1}) + bias_->reshape({1, C, 1, 1});
    }
    return xn;
}

} // namespace pygo::nn

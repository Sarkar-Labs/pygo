#pragma once
#include "pygo/tensor.h"
#include "pygo/nn.h"

namespace pygo::nn {

// ─── Batch Normalization ─────────────────────────────────────────────────────
// Ioffe & Szegedy 2015: normalize activations per mini-batch, then scale+shift.
// Forward (train):  y = (x - μ_B) / √(σ²_B + ε) * γ + β
// Forward (eval):   y = (x - μ_running) / √(σ²_running + ε) * γ + β
class BatchNorm2d : public Module {
public:
    explicit BatchNorm2d(int64_t num_features, float eps = 1e-5f,
                         float momentum = 0.1f, bool affine = true,
                         bool track_running_stats = true);

    Tensor forward(const TensorList& inputs) override;

    // Accessors
    Tensor& weight()          { return *weight_; }
    Tensor& bias()            { return *bias_; }
    Tensor& running_mean()    { return running_mean_; }
    Tensor& running_var()     { return running_var_; }

    void set_training(bool t) { training_ = t; }

private:
    int64_t num_features_;
    float   eps_;
    float   momentum_;
    bool    affine_;
    bool    track_running_stats_;
    bool    training_ = true;
    int64_t num_batches_tracked_ = 0;

    std::shared_ptr<Tensor> weight_;        // γ [C]
    std::shared_ptr<Tensor> bias_;          // β [C]
    Tensor running_mean_;                    // μ [C]
    Tensor running_var_;                     // σ² [C]

    // CUDA-fused BN forward (single-pass mean+var+norm)
    Tensor forward_cuda(const Tensor& x);
    Tensor forward_cpu(const Tensor& x);
};

// ─── Layer Normalization ─────────────────────────────────────────────────────
// Ba et al. 2016: normalize over last D dimensions (not batch).
// No running stats — identical behaviour in train and eval.
class LayerNorm : public Module {
public:
    explicit LayerNorm(std::vector<int64_t> normalized_shape,
                       float eps = 1e-5f, bool elementwise_affine = true);

    Tensor forward(const TensorList& inputs) override;

private:
    std::vector<int64_t> normalized_shape_;
    float   eps_;
    bool    affine_;
    std::shared_ptr<Tensor> weight_;   // γ
    std::shared_ptr<Tensor> bias_;     // β
};

// ─── RMS Norm ────────────────────────────────────────────────────────────────
// Zhang & Sennrich 2019: drop mean centering — only RMS scaling.
// Used in LLaMA, Mistral, Gemma.  y = x / RMS(x) * γ
class RMSNorm : public Module {
public:
    explicit RMSNorm(int64_t dim, float eps = 1e-5f);
    Tensor forward(const TensorList& inputs) override;

private:
    int64_t dim_;
    float   eps_;
    std::shared_ptr<Tensor> weight_;   // γ [dim]
};

// ─── Group Normalization ─────────────────────────────────────────────────────
// Wu & He 2018: split channels into G groups, normalize within each.
// Works with batch size 1 (unlike BN). Popular in detection/segmentation.
class GroupNorm : public Module {
public:
    explicit GroupNorm(int64_t num_groups, int64_t num_channels,
                       float eps = 1e-5f, bool affine = true);
    Tensor forward(const TensorList& inputs) override;

private:
    int64_t num_groups_, num_channels_;
    float   eps_;
    bool    affine_;
    std::shared_ptr<Tensor> weight_, bias_;
};

// ─── Instance Normalization ───────────────────────────────────────────────────
class InstanceNorm2d : public Module {
public:
    explicit InstanceNorm2d(int64_t num_features, float eps = 1e-5f,
                            float momentum = 0.1f, bool affine = false);
    Tensor forward(const TensorList& inputs) override;

private:
    int64_t num_features_;
    float   eps_, momentum_;
    bool    affine_;
    std::shared_ptr<Tensor> weight_, bias_;
    Tensor running_mean_, running_var_;
};

} // namespace pygo::nn

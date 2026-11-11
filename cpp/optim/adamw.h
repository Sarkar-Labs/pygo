#pragma once
#include "pygo/tensor.h"
#include <unordered_map>
#include <memory>

namespace pygo::optim {

// ─── Optimizer base ──────────────────────────────────────────────────────────
class Optimizer {
public:
    virtual ~Optimizer() = default;
    virtual void step()      = 0;
    virtual void zero_grad() = 0;
    virtual void add_param_group(const TensorList& params, float lr) = 0;
    virtual std::string name() const = 0;
};

// ─── Per-param optimizer state ────────────────────────────────────────────────
struct AdamState {
    Tensor exp_avg;      // 1st moment (m)
    Tensor exp_avg_sq;   // 2nd moment (v)
    Tensor max_exp_avg_sq; // for AMSGrad
    int64_t step = 0;
};

// ─── AdamW ────────────────────────────────────────────────────────────────────
// Fused, GPU-accelerated AdamW implementation.
// Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
class AdamW : public Optimizer {
public:
    struct Config {
        float lr           = 1e-3f;
        float beta1        = 0.9f;
        float beta2        = 0.999f;
        float eps          = 1e-8f;
        float weight_decay = 0.01f;
        bool  amsgrad      = false;
        bool  fused        = true;   // use fused CUDA kernel when available
    };

    explicit AdamW(const TensorList& params, Config cfg = {});

    void step() override;
    void zero_grad() override;
    void add_param_group(const TensorList& params, float lr) override;
    std::string name() const override { return "AdamW"; }

    // Adjust LR externally (used by schedulers)
    void   set_lr(float lr)  { cfg_.lr = lr; }
    float  get_lr()    const { return cfg_.lr; }

    // Checkpointing
    std::unordered_map<std::string, Tensor> state_dict() const;
    void load_state_dict(const std::unordered_map<std::string, Tensor>& sd);

private:
    struct ParamGroup {
        TensorList params;
        float      lr;
    };

    Config                                   cfg_;
    std::vector<ParamGroup>                  groups_;
    std::unordered_map<TensorImpl*, AdamState> state_;
    int64_t                                  global_step_ = 0;

    // CPU fallback path
    void step_cpu(ParamGroup& g);
    // Fused CUDA path (single kernel launch updates all params)
    void step_cuda_fused(ParamGroup& g);
};

// ─── Lion ────────────────────────────────────────────────────────────────────
// EvoLved Sign Momentum — Chen et al. 2023 (Google Brain symbolic search)
class Lion : public Optimizer {
public:
    struct Config {
        float lr           = 1e-4f;   // Lion typically uses 3-10x smaller LR than AdamW
        float beta1        = 0.9f;
        float beta2        = 0.99f;
        float weight_decay = 0.0f;
    };

    explicit Lion(const TensorList& params, Config cfg = {});
    void step() override;
    void zero_grad() override;
    void add_param_group(const TensorList& params, float lr) override;
    std::string name() const override { return "Lion"; }

private:
    Config cfg_;
    struct ParamGroup { TensorList params; float lr; };
    std::vector<ParamGroup> groups_;
    std::unordered_map<TensorImpl*, Tensor> momentum_;
};

// ─── Sophia ──────────────────────────────────────────────────────────────────
// Second-order clipped stochastic optimizer (Liu et al. 2023)
// Uses diagonal Hessian estimate (Hutchinson or Gauss-Newton-Bartlett)
class Sophia : public Optimizer {
public:
    struct Config {
        float lr           = 1e-4f;
        float beta1        = 0.965f;
        float beta2        = 0.99f;
        float rho          = 0.04f;   // clip threshold
        float weight_decay = 0.1f;
        int64_t hessian_update_freq = 10;
    };

    explicit Sophia(const TensorList& params, Config cfg = {});
    void step() override;
    void zero_grad() override;
    void add_param_group(const TensorList& params, float lr) override;
    std::string name() const override { return "Sophia"; }

    // Call periodically to refresh diagonal Hessian estimate
    void update_hessian(const TensorList& gauss_newton_bartlett_grads);

private:
    Config cfg_;
    struct ParamGroup { TensorList params; float lr; };
    std::vector<ParamGroup> groups_;
    struct SophiaState {
        Tensor exp_avg;       // m
        Tensor hessian_diag;  // h (EMA of diagonal Hessian estimate)
        int64_t step = 0;
    };
    std::unordered_map<TensorImpl*, SophiaState> state_;
    int64_t global_step_ = 0;
};

// ─── LAMB ────────────────────────────────────────────────────────────────────
// Layer-wise Adaptive Moments for Batch training (You et al. 2020)
// Used for very large batch sizes (32K+) in BERT/ResNet pretraining
class LAMB : public Optimizer {
public:
    struct Config {
        float lr           = 1e-3f;
        float beta1        = 0.9f;
        float beta2        = 0.999f;
        float eps          = 1e-6f;
        float weight_decay = 0.01f;
        float trust_clip   = 10.0f;  // clamp trust ratio
    };

    explicit LAMB(const TensorList& params, Config cfg = {});
    void step() override;
    void zero_grad() override;
    void add_param_group(const TensorList& params, float lr) override;
    std::string name() const override { return "LAMB"; }

private:
    Config cfg_;
    struct ParamGroup { TensorList params; float lr; };
    std::vector<ParamGroup> groups_;
    std::unordered_map<TensorImpl*, AdamState> state_;
    int64_t global_step_ = 0;
};

// ─── Factory ─────────────────────────────────────────────────────────────────
std::unique_ptr<Optimizer> make_optimizer(
    const std::string& name, const TensorList& params,
    float lr, float weight_decay = 0.01f);

} // namespace pygo::optim

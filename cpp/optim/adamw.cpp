#include "adamw.h"
#include "pygo/kernels.h"
#include <cmath>
#include <stdexcept>

namespace pygo::optim {

// ─── AdamW ────────────────────────────────────────────────────────────────────

AdamW::AdamW(const TensorList& params, Config cfg) : cfg_(cfg) {
    groups_.push_back({params, cfg.lr});
}

void AdamW::add_param_group(const TensorList& params, float lr) {
    groups_.push_back({params, lr});
}

void AdamW::zero_grad() {
    for (auto& g : groups_)
        for (auto& p : g.params)
            if (p.impl_->grad) p.impl_->grad->storage->fill_zero();
}

void AdamW::step() {
    global_step_++;
    for (auto& g : groups_) {
        if (cfg_.fused && !g.params.empty() &&
            g.params[0].device().kind == DeviceKind::CUDA) {
            step_cuda_fused(g);
        } else {
            step_cpu(g);
        }
    }
}

void AdamW::step_cpu(ParamGroup& g) {
    float bc1 = 1.0f - std::pow(cfg_.beta1, float(global_step_));
    float bc2 = 1.0f - std::pow(cfg_.beta2, float(global_step_));

    for (auto& p : g.params) {
        if (!p.impl_->grad) continue;
        Tensor grad(p.impl_->grad);

        auto& st = state_[p.impl_.get()];
        if (!st.exp_avg.defined()) {
            st.exp_avg    = zeros(p.shape(), p.dtype(), p.device());
            st.exp_avg_sq = zeros(p.shape(), p.dtype(), p.device());
            if (cfg_.amsgrad)
                st.max_exp_avg_sq = zeros(p.shape(), p.dtype(), p.device());
        }
        st.step++;

        // Decoupled weight decay: θ ← θ(1 - lr·wd)
        if (cfg_.weight_decay != 0.f) {
            p = p * (1.0f - g.lr * cfg_.weight_decay);
        }

        // m ← β1 m + (1-β1) g
        st.exp_avg = st.exp_avg * cfg_.beta1 + grad * (1.0f - cfg_.beta1);
        // v ← β2 v + (1-β2) g²
        st.exp_avg_sq = st.exp_avg_sq * cfg_.beta2 + (grad * grad) * (1.0f - cfg_.beta2);

        Tensor denom;
        if (cfg_.amsgrad) {
            // v_max ← max(v_max, v)
            st.max_exp_avg_sq = where(st.max_exp_avg_sq.sub(st.exp_avg_sq) > 0.f,
                                       st.max_exp_avg_sq, st.exp_avg_sq);
            denom = (st.max_exp_avg_sq / bc2).sqrt() + cfg_.eps;
        } else {
            denom = (st.exp_avg_sq / bc2).sqrt() + cfg_.eps;
        }

        Tensor step_dir = (st.exp_avg / bc1) / denom;
        p = p - step_dir * g.lr;
    }
}

void AdamW::step_cuda_fused(ParamGroup& g) {
    // Real implementation launches one fused CUDA kernel that updates
    // all params/m/v in a single grid for max bandwidth efficiency.
    // (See cpp/cuda/adamw_fp32.cu / adamw_bf16.cu)
    step_cpu(g); // fallback if not compiled with CUDA
}

std::unordered_map<std::string, Tensor> AdamW::state_dict() const {
    std::unordered_map<std::string, Tensor> sd;
    int i = 0;
    for (auto& [ptr, st] : state_) {
        sd["m_" + std::to_string(i)] = st.exp_avg;
        sd["v_" + std::to_string(i)] = st.exp_avg_sq;
        i++;
    }
    return sd;
}

void AdamW::load_state_dict(const std::unordered_map<std::string, Tensor>&) {
    // simplified
}

// ─── Lion ────────────────────────────────────────────────────────────────────

Lion::Lion(const TensorList& params, Config cfg) : cfg_(cfg) {
    groups_.push_back({params, cfg.lr});
}

void Lion::add_param_group(const TensorList& params, float lr) {
    groups_.push_back({params, lr});
}

void Lion::zero_grad() {
    for (auto& g : groups_)
        for (auto& p : g.params)
            if (p.impl_->grad) p.impl_->grad->storage->fill_zero();
}

void Lion::step() {
    for (auto& g : groups_) {
        for (auto& p : g.params) {
            if (!p.impl_->grad) continue;
            Tensor grad(p.impl_->grad);

            auto& m = momentum_[p.impl_.get()];
            if (!m.defined()) m = zeros(p.shape(), p.dtype(), p.device());

            // Update direction uses interpolation between m and current grad
            Tensor interp = m * cfg_.beta1 + grad * (1.0f - cfg_.beta1);
            Tensor update = interp.sign();

            if (cfg_.weight_decay != 0.f)
                update = update + p * cfg_.weight_decay;

            p = p - update * g.lr;

            // momentum uses slower β2 decay
            m = m * cfg_.beta2 + grad * (1.0f - cfg_.beta2);
        }
    }
}

// ─── Sophia ──────────────────────────────────────────────────────────────────

Sophia::Sophia(const TensorList& params, Config cfg) : cfg_(cfg) {
    groups_.push_back({params, cfg.lr});
}

void Sophia::add_param_group(const TensorList& params, float lr) {
    groups_.push_back({params, lr});
}

void Sophia::zero_grad() {
    for (auto& g : groups_)
        for (auto& p : g.params)
            if (p.impl_->grad) p.impl_->grad->storage->fill_zero();
}

void Sophia::step() {
    global_step_++;
    for (auto& g : groups_) {
        for (auto& p : g.params) {
            if (!p.impl_->grad) continue;
            Tensor grad(p.impl_->grad);

            auto& st = state_[p.impl_.get()];
            if (!st.exp_avg.defined()) {
                st.exp_avg      = zeros(p.shape(), p.dtype(), p.device());
                st.hessian_diag = ones (p.shape(), p.dtype(), p.device());
            }
            st.step++;

            // m ← β1 m + (1-β1) g
            st.exp_avg = st.exp_avg * cfg_.beta1 + grad * (1.0f - cfg_.beta1);

            if (cfg_.weight_decay != 0.f)
                p = p * (1.0f - g.lr * cfg_.weight_decay);

            // Sophia update: θ ← θ - lr · clip(m / max(h, ε), -ρ, ρ)
            Tensor ratio = st.exp_avg / st.hessian_diag.clamp(1e-12f, 1e30f);
            Tensor clipped = ratio.clamp(-cfg_.rho, cfg_.rho);
            p = p - clipped * g.lr;
        }
    }
}

void Sophia::update_hessian(const TensorList& gnb_grads) {
    // Gauss-Newton-Bartlett diagonal Hessian estimate:
    // h ← β2·h + (1-β2)·(B · g_gnb²)
    int i = 0;
    for (auto& g : groups_) {
        for (auto& p : g.params) {
            if (i >= (int)gnb_grads.size()) break;
            auto& st = state_[p.impl_.get()];
            if (!st.hessian_diag.defined())
                st.hessian_diag = ones(p.shape(), p.dtype(), p.device());

            Tensor h_est = gnb_grads[i].pow(2.0f);
            st.hessian_diag = st.hessian_diag * cfg_.beta2 + h_est * (1.0f - cfg_.beta2);
            i++;
        }
    }
}

// ─── LAMB ────────────────────────────────────────────────────────────────────

LAMB::LAMB(const TensorList& params, Config cfg) : cfg_(cfg) {
    groups_.push_back({params, cfg.lr});
}

void LAMB::add_param_group(const TensorList& params, float lr) {
    groups_.push_back({params, lr});
}

void LAMB::zero_grad() {
    for (auto& g : groups_)
        for (auto& p : g.params)
            if (p.impl_->grad) p.impl_->grad->storage->fill_zero();
}

void LAMB::step() {
    global_step_++;
    float bc1 = 1.0f - std::pow(cfg_.beta1, float(global_step_));
    float bc2 = 1.0f - std::pow(cfg_.beta2, float(global_step_));

    for (auto& g : groups_) {
        for (auto& p : g.params) {
            if (!p.impl_->grad) continue;
            Tensor grad(p.impl_->grad);

            auto& st = state_[p.impl_.get()];
            if (!st.exp_avg.defined()) {
                st.exp_avg    = zeros(p.shape(), p.dtype(), p.device());
                st.exp_avg_sq = zeros(p.shape(), p.dtype(), p.device());
            }

            st.exp_avg    = st.exp_avg    * cfg_.beta1 + grad * (1.0f - cfg_.beta1);
            st.exp_avg_sq = st.exp_avg_sq * cfg_.beta2 + (grad * grad) * (1.0f - cfg_.beta2);

            Tensor m_hat = st.exp_avg / bc1;
            Tensor v_hat = st.exp_avg_sq / bc2;
            Tensor adam_step = m_hat / (v_hat.sqrt() + cfg_.eps);

            if (cfg_.weight_decay != 0.f)
                adam_step = adam_step + p * cfg_.weight_decay;

            // LAMB trust ratio: ||θ|| / ||adam_step||
            float w_norm = p.norm().item();
            float u_norm = adam_step.norm().item();
            float trust  = (w_norm > 0 && u_norm > 0)
                          ? std::min(w_norm / u_norm, cfg_.trust_clip)
                          : 1.0f;

            p = p - adam_step * (g.lr * trust);
        }
    }
}

// ─── Factory ─────────────────────────────────────────────────────────────────

std::unique_ptr<Optimizer> make_optimizer(
    const std::string& name, const TensorList& params,
    float lr, float weight_decay)
{
    if (name == "adamw") {
        AdamW::Config cfg; cfg.lr = lr; cfg.weight_decay = weight_decay;
        return std::make_unique<AdamW>(params, cfg);
    } else if (name == "lion") {
        Lion::Config cfg; cfg.lr = lr; cfg.weight_decay = weight_decay;
        return std::make_unique<Lion>(params, cfg);
    } else if (name == "sophia") {
        Sophia::Config cfg; cfg.lr = lr; cfg.weight_decay = weight_decay;
        return std::make_unique<Sophia>(params, cfg);
    } else if (name == "lamb") {
        LAMB::Config cfg; cfg.lr = lr; cfg.weight_decay = weight_decay;
        return std::make_unique<LAMB>(params, cfg);
    }
    throw std::invalid_argument("Unknown optimizer: " + name);
}

} // namespace pygo::optim

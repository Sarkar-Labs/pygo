#pragma once
#include "pygo/tensor.h"
#include "pygo/nn.h"
#include "linear.h"
#include "dropout.h"
#include <optional>
#include <variant>

namespace pygo::nn {

// ─── AttentionConfig ─────────────────────────────────────────────────────────
struct AttentionConfig {
    int64_t d_model      = 512;
    int64_t n_heads      = 8;
    int64_t n_kv_heads   = 8;       // for GQA; equals n_heads for MHA
    float   dropout      = 0.0f;
    bool    causal       = false;
    bool    use_flash    = true;
    bool    use_rope     = true;
    float   rope_theta   = 500000.f;
    int64_t max_seq_len  = 4096;
    std::optional<int64_t> sliding_window;
    float   alibi_slope  = 0.0f;
    bool    qkv_bias     = false;
    bool    out_bias     = false;

    int64_t head_dim()   const { return d_model / n_heads; }
    int64_t n_q_per_kv() const { return n_heads / n_kv_heads; }
};

// ─── KVCache ─────────────────────────────────────────────────────────────────
struct KVCache {
    Tensor k;   // [B, H_kv, S_past, D]
    Tensor v;   // [B, H_kv, S_past, D]
    int64_t seen_tokens = 0;

    bool   empty() const { return seen_tokens == 0; }
    void   reset() { seen_tokens = 0; }
    void   update(const Tensor& new_k, const Tensor& new_v);

    // Paged KV cache support
    struct PageTable {
        std::vector<int32_t> page_ids;
        int32_t              block_size = 16;
    };
    std::optional<PageTable> pages;
};

// ─── RoPE ────────────────────────────────────────────────────────────────────
class RotaryEmbedding : public Module {
public:
    explicit RotaryEmbedding(int64_t dim, int64_t max_seq, float theta = 500000.f);

    // Pre-compute cos/sin tables
    void build_cache(int64_t seq_len);

    // Apply RoPE to query and key tensors in-place
    // q,k: [B, H, S, D]  →  rotated in-place
    void apply(Tensor& q, Tensor& k,
               const std::optional<Tensor>& pos_ids = std::nullopt);

    Tensor forward(const TensorList& inputs) override;

private:
    int64_t dim_, max_seq_;
    float   theta_;
    Tensor  cos_cache_;  // [max_seq, dim/2]
    Tensor  sin_cache_;  // [max_seq, dim/2]

    static Tensor rotate_half(const Tensor& x);
};

// ─── Multi-Head Attention ────────────────────────────────────────────────────
class MultiHeadAttention : public Module {
public:
    explicit MultiHeadAttention(const AttentionConfig& cfg);

    // x: [B, S, D]
    // mask: optional [1, 1, S, S] additive mask
    // Returns: (output [B,S,D], optional updated cache)
    std::pair<Tensor, std::optional<KVCache>>
    forward_with_cache(
        const Tensor& x,
        const std::optional<Tensor>& mask   = std::nullopt,
        const std::optional<Tensor>& pos_ids= std::nullopt,
        std::optional<KVCache>       cache  = std::nullopt,
        bool                         use_cache = false);

    Tensor forward(const TensorList& inputs) override;

    // Expose sub-modules
    Linear& q_proj() { return *q_proj_; }
    Linear& k_proj() { return *k_proj_; }
    Linear& v_proj() { return *v_proj_; }
    Linear& o_proj() { return *o_proj_; }

private:
    friend class CrossAttention;
    AttentionConfig cfg_;
    std::shared_ptr<Linear>          q_proj_;
    std::shared_ptr<Linear>          k_proj_;
    std::shared_ptr<Linear>          v_proj_;
    std::shared_ptr<Linear>          o_proj_;
    std::shared_ptr<RotaryEmbedding> rope_;
    std::shared_ptr<Dropout>         attn_drop_;

    // Expand KV heads for GQA
    static Tensor repeat_kv(const Tensor& x, int64_t n_rep);

    // Build causal mask [1,1,S,S]
    static Tensor causal_mask(int64_t seq_len, Device dev);

    // Core attention computation (dispatch to Flash or SDPA)
    Tensor compute_attention(
        const Tensor& q,      // [B, H, S, D]
        const Tensor& k,      // [B, H_kv, S, D]
        const Tensor& v,      // [B, H_kv, S, D]
        const std::optional<Tensor>& mask,
        float scale);
};

// ─── Flash Attention wrapper ─────────────────────────────────────────────────
// Thin C++ wrapper around the CUDA .cu kernel
Tensor flash_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    bool causal      = false,
    float scale      = -1.f,
    float alibi_slope= 0.f);

// ─── Scaled Dot-Product Attention (CPU/fp32 fallback) ───────────────────────
Tensor scaled_dot_product_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const std::optional<Tensor>& mask = std::nullopt,
    float dropout_p = 0.f,
    bool  causal    = false,
    float scale     = -1.f);

// ─── Cross-Attention ────────────────────────────────────────────────────────
class CrossAttention : public MultiHeadAttention {
public:
    using MultiHeadAttention::MultiHeadAttention;

    // query from decoder, key/value from encoder
    Tensor cross_forward(
        const Tensor& query,
        const Tensor& context,
        const std::optional<Tensor>& mask = std::nullopt);
};

// ─── Sliding Window Attention ────────────────────────────────────────────────
class SlidingWindowAttention : public Module {
public:
    SlidingWindowAttention(const AttentionConfig& cfg, int64_t window_size);
    Tensor forward(const TensorList& inputs) override;
private:
    MultiHeadAttention base_;
    int64_t window_size_;
    Tensor  build_sliding_mask(int64_t seq_len, Device dev);
};

// ─── ALiBi Bias ─────────────────────────────────────────────────────────────
Tensor alibi_bias(int64_t n_heads, int64_t seq_len, Device dev);

} // namespace pygo::nn

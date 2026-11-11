#include "attention.h"
#include "pygo/kernels.h"
#include "pygo/softmax.h"
#include "pygo/device.h"
#include <cmath>
#include <stdexcept>

namespace pygo::nn {

// ─── RotaryEmbedding ────────────────────────────────────────────────────────

RotaryEmbedding::RotaryEmbedding(int64_t dim, int64_t max_seq, float theta)
    : dim_(dim), max_seq_(max_seq), theta_(theta) {
    build_cache(max_seq);
}

void RotaryEmbedding::build_cache(int64_t seq_len) {
    // inv_freq: [dim/2]
    auto half = dim_ / 2;
    auto t = arange(0, float(seq_len), 1.f);   // [S]
    auto inv_freq_vals = std::vector<float>(half);
    for (int i = 0; i < half; ++i)
        inv_freq_vals[i] = 1.0f / std::pow(theta_, float(2*i) / float(dim_));
    auto inv_freq = Tensor::from_data(inv_freq_vals.data(), Shape{half}, DType::Float32, Device::CPU);

    // freqs [S, half] = outer(t, inv_freq)
    auto freqs = t.unsqueeze(1).matmul(inv_freq.unsqueeze(0)); // [S, half]
    auto emb   = cat({freqs, freqs}, -1);                      // [S, dim]
    cos_cache_ = emb.cos();  // [S, dim]
    sin_cache_ = emb.sin();  // [S, dim]
    max_seq_   = seq_len;
}

Tensor RotaryEmbedding::rotate_half(const Tensor& x) {
    // x[..., :D/2]  x[..., D/2:]  → [-x2, x1]
    auto half = x.dim(-1) / 2;
    auto x1   = x.slice(-1, 0, half);
    auto x2   = x.slice(-1, half, x.dim(-1));
    return cat({x2.neg(), x1}, -1);
}

void RotaryEmbedding::apply(Tensor& q, Tensor& k,
                             const std::optional<Tensor>& pos_ids) {
    int64_t seq = q.dim(-2);
    if (seq > max_seq_) build_cache(seq * 2);

    Tensor cos, sin;
    if (pos_ids.has_value()) {
        // Advanced tensor indexing is not part of the stable CPU tensor ABI yet.
        // Do not silently produce the wrong positional encoding.
        throw std::runtime_error("RotaryEmbedding::apply: position-id indexing is not supported by the current tensor ABI");
    } else {
        cos = cos_cache_.slice(0, 0, seq).unsqueeze(0).unsqueeze(0);
        sin = sin_cache_.slice(0, 0, seq).unsqueeze(0).unsqueeze(0);
    }

    q = (q * cos) + (rotate_half(q) * sin);
    k = (k * cos.slice(-2, 0, k.dim(-2)))
      + (rotate_half(k) * sin.slice(-2, 0, k.dim(-2)));
}

Tensor RotaryEmbedding::forward(const TensorList& inputs) {
    throw std::runtime_error("Use apply() directly");
}

// ─── KVCache update ──────────────────────────────────────────────────────────

void KVCache::update(const Tensor& new_k, const Tensor& new_v) {
    if (empty()) {
        k = new_k;
        v = new_v;
    } else {
        k = cat({k, new_k}, 2);
        v = cat({v, new_v}, 2);
    }
    seen_tokens += new_k.dim(2);
}

// ─── MultiHeadAttention ──────────────────────────────────────────────────────

MultiHeadAttention::MultiHeadAttention(const AttentionConfig& cfg) : cfg_(cfg) {
    auto d   = cfg.d_model;
    auto hd  = cfg.head_dim();
    auto nkv = cfg.n_kv_heads;
    auto nh  = cfg.n_heads;

    q_proj_    = std::make_shared<Linear>(d, nh  * hd, cfg.qkv_bias);
    k_proj_    = std::make_shared<Linear>(d, nkv * hd, cfg.qkv_bias);
    v_proj_    = std::make_shared<Linear>(d, nkv * hd, cfg.qkv_bias);
    o_proj_    = std::make_shared<Linear>(nh * hd, d,  cfg.out_bias);
    attn_drop_ = std::make_shared<Dropout>(cfg.dropout);

    if (cfg.use_rope) {
        rope_ = std::make_shared<RotaryEmbedding>(
            hd, cfg.max_seq_len, cfg.rope_theta);
    }

    register_module("q_proj",    q_proj_);
    register_module("k_proj",    k_proj_);
    register_module("v_proj",    v_proj_);
    register_module("o_proj",    o_proj_);
    register_module("attn_drop", attn_drop_);
    if (rope_) register_module("rope", rope_);
}

Tensor MultiHeadAttention::repeat_kv(const Tensor& x, int64_t n_rep) {
    if (n_rep == 1) return x;
    auto [B, H, S, D] = x.shape().as_tuple4();
    return x.unsqueeze(2)
             .expand({B, H, n_rep, S, D})
             .reshape({B, H * n_rep, S, D});
}

Tensor MultiHeadAttention::causal_mask(int64_t seq_len, Device dev) {
    auto mask = full(Shape{seq_len, seq_len}, -1e38f, DType::Float32, dev);
    // zero upper triangle (keep lower)
    return mask.triu(1).unsqueeze(0).unsqueeze(0);
}

Tensor MultiHeadAttention::compute_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const std::optional<Tensor>& mask, float scale)
{
    if (cfg_.use_flash && q.device().kind == DeviceKind::CUDA) {
        return flash_attention(q, k, v, cfg_.causal, scale, cfg_.alibi_slope);
    }
    return scaled_dot_product_attention(
        q, k, v, mask, cfg_.dropout, cfg_.causal, scale);
}

std::pair<Tensor, std::optional<KVCache>>
MultiHeadAttention::forward_with_cache(
    const Tensor& x,
    const std::optional<Tensor>& mask,
    const std::optional<Tensor>& pos_ids,
    std::optional<KVCache>       cache,
    bool use_cache)
{
    auto [B, S, _] = x.shape().as_tuple3();
    auto hd  = cfg_.head_dim();
    auto nh  = cfg_.n_heads;
    auto nkv = cfg_.n_kv_heads;

    auto q = q_proj_->forward({x}).view({B, S, nh,  hd}).transpose(1, 2);
    auto k = k_proj_->forward({x}).view({B, S, nkv, hd}).transpose(1, 2);
    auto v = v_proj_->forward({x}).view({B, S, nkv, hd}).transpose(1, 2);

    // Apply RoPE
    if (rope_) rope_->apply(q, k, pos_ids);

    // Concat with KV cache
    if (cache.has_value()) {
        cache->update(k, v);
        k = cache->k;
        v = cache->v;
    }
    std::optional<KVCache> new_cache;
    if (use_cache) {
        new_cache = KVCache{k, v, k.dim(2)};
    }

    // GQA: repeat KV heads
    k = repeat_kv(k, cfg_.n_q_per_kv());
    v = repeat_kv(v, cfg_.n_q_per_kv());

    // Build mask if not provided and causal
    std::optional<Tensor> attn_mask = mask;
    if (!attn_mask.has_value() && cfg_.causal && !cfg_.use_flash) {
        attn_mask = causal_mask(k.dim(2), x.device());
    }

    float scale = 1.0f / std::sqrt(float(hd));
    auto out = compute_attention(q, k, v, attn_mask, scale);

    out = out.transpose(1, 2).reshape({B, S, nh * hd});
    out = o_proj_->forward({out});
    return {out, new_cache};
}

Tensor MultiHeadAttention::forward(const TensorList& inputs) {
    auto [out, _] = forward_with_cache(inputs[0]);
    return out;
}

// ─── Scaled Dot-Product Attention ────────────────────────────────────────────

Tensor scaled_dot_product_attention(
    const Tensor& q, const Tensor& k, const Tensor& v,
    const std::optional<Tensor>& mask,
    float dropout_p, bool causal, float scale)
{
    int64_t D = q.dim(-1);
    if (scale < 0) scale = 1.0f / std::sqrt(float(D));

    // scores = q @ k.T * scale   [B, H, Sq, Sk]
    auto scores = q.matmul(k.transpose(-2, -1)) * scale;

    if (causal && !mask.has_value()) {
        int64_t Sq = q.dim(-2), Sk = k.dim(-2);
        auto cm = full(Shape{Sq, Sk}, -1e38f, DType::Float32, q.device()).triu(1);
        scores = scores + cm.unsqueeze(0).unsqueeze(0);
    } else if (mask.has_value()) {
        scores = scores + *mask;
    }

    auto attn = softmax(scores, -1).to(q.dtype());

    if (dropout_p > 0.f) {
        // training dropout applied to attention weights
        attn = Dropout(dropout_p).forward({attn});
    }

    return attn.matmul(v);
}

// ─── ALiBi ───────────────────────────────────────────────────────────────────

Tensor alibi_bias(int64_t n_heads, int64_t seq_len, Device dev) {
    // slopes: 2^(−8/n_heads)  for each head
    std::vector<float> slopes(n_heads);
    for (int h = 0; h < n_heads; ++h)
        slopes[h] = std::pow(2.f, -8.f * (h + 1) / float(n_heads));

    auto pos  = arange(0, float(seq_len), 1.f, dev);           // [S]
    auto bias = pos.unsqueeze(0) - pos.unsqueeze(1);           // [S, S]
    auto sl   = Tensor::from_data(slopes.data(), Shape{n_heads, 1, 1}, DType::Float32, dev);
    return bias.unsqueeze(0) * sl;  // [H, S, S]
}

// ─── CrossAttention ──────────────────────────────────────────────────────────

Tensor CrossAttention::cross_forward(
    const Tensor& query, const Tensor& context,
    const std::optional<Tensor>& mask)
{
    auto [B, Sq, _q] = query.shape().as_tuple3();
    auto hd  = cfg_.head_dim();
    auto nh  = cfg_.n_heads;
    auto nkv = cfg_.n_kv_heads;

    auto q = q_proj_->forward({query})  .view({B, Sq, nh,  hd}).transpose(1, 2);
    auto k = k_proj_->forward({context}).view({B, -1, nkv, hd}).transpose(1, 2);
    auto v = v_proj_->forward({context}).view({B, -1, nkv, hd}).transpose(1, 2);

    k = repeat_kv(k, cfg_.n_q_per_kv());
    v = repeat_kv(v, cfg_.n_q_per_kv());

    float scale = 1.0f / std::sqrt(float(hd));
    auto out = compute_attention(q, k, v, mask, scale);
    out = out.transpose(1, 2).reshape({B, Sq, nh * hd});
    return o_proj_->forward({out});
}

} // namespace pygo::nn

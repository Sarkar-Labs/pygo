"""PyGo Transformer — full GPT/BERT/T5/LLaMA-style implementation."""
from __future__ import annotations
import math
import torch
from typing import Optional, Tuple
from dataclasses import dataclass, field
from .module    import Module
from .linear    import Linear
from .layer_norm import LayerNorm
from .dropout   import Dropout
from .embedding import Embedding
from ..tensor   import Tensor


# ─── Config ──────────────────────────────────────────────────────────────────
@dataclass
class TransformerConfig:
    vocab_size:    int   = 32_000
    max_seq_len:   int   = 4_096
    d_model:       int   = 4_096
    n_heads:       int   = 32
    n_kv_heads:    int   = 8          # GQA: grouped-query attention
    n_layers:      int   = 32
    d_ff:          int   = 14_336     # SwiGLU → 8/3 * d_model rounded
    dropout:       float = 0.0
    attn_dropout:  float = 0.0
    rope_theta:    float = 500_000.0  # LLaMA-3 uses 500k
    rope_scaling:  Optional[dict] = None
    norm_eps:      float = 1e-5
    tie_word_emb:  bool  = True
    use_flash_attn:bool  = True
    use_gqa:       bool  = True       # Grouped-Query Attention
    use_swiglu:    bool  = True
    use_rms_norm:  bool  = True
    use_alibi:     bool  = False
    sliding_window:Optional[int] = None
    pad_token_id:  int   = 0
    bos_token_id:  int   = 1
    eos_token_id:  int   = 2

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    @property
    def n_q_per_kv(self) -> int:
        return self.n_heads // self.n_kv_heads


# ─── RMS Norm ────────────────────────────────────────────────────────────────
class RMSNorm(Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps   = eps
        self.weight = torch.nn.Parameter(torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# ─── RoPE Embeddings ─────────────────────────────────────────────────────────
class RotaryEmbedding(Module):
    def __init__(self, dim: int, max_seq_len: int, theta: float = 500_000.0,
                 scaling: Optional[dict] = None):
        super().__init__()
        self.dim = dim
        self.theta = theta
        self.scaling = scaling
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t     = torch.arange(seq_len, device=self.inv_freq.device).float()
        freqs = torch.outer(t, self.inv_freq)
        emb   = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer("cos_cached", emb.cos()[None, None, :, :])
        self.register_buffer("sin_cached", emb.sin()[None, None, :, :])

    @staticmethod
    def _rotate_half(x: Tensor) -> Tensor:
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: Tensor, k: Tensor,
                position_ids: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        seq_len = q.shape[-2]
        if seq_len > self.cos_cached.shape[-2]:
            self._build_cache(seq_len * 2)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        if position_ids is not None:
            cos = cos.squeeze()[position_ids].unsqueeze(1)
            sin = sin.squeeze()[position_ids].unsqueeze(1)

        q_rot = (q * cos) + (self._rotate_half(q) * sin)
        k_rot = (k * cos[:, :, :k.shape[-2], :]) + \
                (self._rotate_half(k) * sin[:, :, :k.shape[-2], :])
        return q_rot, k_rot


# ─── Grouped-Query Attention ─────────────────────────────────────────────────
class GroupedQueryAttention(Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg      = cfg
        self.n_heads  = cfg.n_heads
        self.n_kv     = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        self.scale    = self.head_dim ** -0.5
        self.n_rep    = cfg.n_q_per_kv  # queries per kv head

        self.q_proj   = Linear(cfg.d_model, cfg.n_heads  * cfg.head_dim, bias=False)
        self.k_proj   = Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.v_proj   = Linear(cfg.d_model, cfg.n_kv_heads * cfg.head_dim, bias=False)
        self.o_proj   = Linear(cfg.n_heads * cfg.head_dim, cfg.d_model,    bias=False)
        self.attn_drop= Dropout(cfg.attn_dropout)
        self.rope     = RotaryEmbedding(cfg.head_dim, cfg.max_seq_len, cfg.rope_theta,
                                         cfg.rope_scaling)

    @staticmethod
    def _repeat_kv(x: Tensor, n_rep: int) -> Tensor:
        """Expand KV heads to match Q heads."""
        if n_rep == 1:
            return x
        B, H, S, D = x.shape
        return x[:, :, None, :, :].expand(B, H, n_rep, S, D) \
                .reshape(B, H * n_rep, S, D)

    def forward(self, x: Tensor,
                mask: Optional[Tensor]       = None,
                position_ids: Optional[Tensor] = None,
                kv_cache: Optional[Tuple[Tensor, Tensor]] = None,
                use_cache: bool = False
                ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.n_heads,  self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv,     self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv,     self.head_dim).transpose(1, 2)

        # Apply RoPE
        q, k = self.rope(q, k, position_ids)

        # KV Cache concat
        if kv_cache is not None:
            k = torch.cat([kv_cache[0], k], dim=2)
            v = torch.cat([kv_cache[1], v], dim=2)
        new_cache = (k, v) if use_cache else None

        # GQA: repeat KV heads
        k = self._repeat_kv(k, self.n_rep)
        v = self._repeat_kv(v, self.n_rep)

        # Attention
        if self.cfg.use_flash_attn:
            from ..tensor import flash_attention
            out = flash_attention(q, k, v, causal=mask is None,
                                  scale=self.scale)
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            if mask is not None:
                scores = scores + mask
            attn   = torch.softmax(scores.float(), dim=-1).to(q.dtype)
            attn   = self.attn_drop(attn)
            out    = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(B, S, -1)
        out = self.o_proj(out)
        return out, new_cache


# ─── SwiGLU FFN ──────────────────────────────────────────────────────────────
class SwiGLU(Module):
    """SwiGLU: out = silu(W1·x) ⊙ W3·x, then W2·that"""
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate_proj = Linear(d_model, d_ff, bias=False)
        self.up_proj   = Linear(d_model, d_ff, bias=False)
        self.down_proj = Linear(d_ff,   d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(
            torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x)
        )


# ─── Transformer Layer ───────────────────────────────────────────────────────
class TransformerLayer(Module):
    def __init__(self, cfg: TransformerConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        norm_cls = RMSNorm if cfg.use_rms_norm else LayerNorm
        self.input_norm  = norm_cls(cfg.d_model, cfg.norm_eps)
        self.post_norm   = norm_cls(cfg.d_model, cfg.norm_eps)
        self.attn        = GroupedQueryAttention(cfg)
        self.ffn         = SwiGLU(cfg.d_model, cfg.d_ff) if cfg.use_swiglu \
                           else _GPTFFN(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop        = Dropout(cfg.dropout)

    def forward(self, x: Tensor, mask=None, position_ids=None,
                kv_cache=None, use_cache=False):
        # Pre-norm attention with residual
        residual = x
        x        = self.input_norm(x)
        x, cache = self.attn(x, mask=mask, position_ids=position_ids,
                              kv_cache=kv_cache, use_cache=use_cache)
        x        = self.drop(x) + residual

        # Pre-norm FFN with residual
        residual = x
        x        = self.ffn(self.post_norm(x))
        x        = self.drop(x) + residual
        return x, cache


class _GPTFFN(Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1  = Linear(d_model, d_ff)
        self.fc2  = Linear(d_ff, d_model)
        self.drop = Dropout(dropout)
    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(self.drop(torch.nn.functional.gelu(self.fc1(x))))


# ─── Full LLaMA-style Transformer ────────────────────────────────────────────
class CausalTransformer(Module):
    """Auto-regressive decoder-only transformer (LLaMA / GPT / Mistral style)."""
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg     = cfg
        self.embed   = Embedding(cfg.vocab_size, cfg.d_model)
        self.drop    = Dropout(cfg.dropout)
        self.layers  = torch.nn.ModuleList(
            [TransformerLayer(cfg, i) for i in range(cfg.n_layers)]
        )
        norm_cls     = RMSNorm if cfg.use_rms_norm else LayerNorm
        self.norm    = norm_cls(cfg.d_model, cfg.norm_eps)
        self.lm_head = Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_word_emb:
            self.lm_head.weight = self.embed.weight

        self._init_weights()

    def _init_weights(self):
        std = self.cfg.d_model ** -0.5
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.normal_(p, std=std)

    def _build_causal_mask(self, seq_len: int, device) -> Tensor:
        mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :]  # [1,1,S,S]

    def forward(self, input_ids: Tensor,
                attention_mask: Optional[Tensor]     = None,
                position_ids:   Optional[Tensor]     = None,
                past_key_values: Optional[list]      = None,
                use_cache: bool                       = False,
                output_hidden_states: bool            = False
                ) -> dict:
        B, S        = input_ids.shape
        device      = input_ids.device
        past_len    = past_key_values[0][0].shape[2] if past_key_values else 0

        if position_ids is None:
            position_ids = torch.arange(past_len, past_len + S,
                                        device=device).unsqueeze(0)

        x = self.drop(self.embed(input_ids))

        causal_mask = self._build_causal_mask(S + past_len, device)

        all_hidden, all_caches = [], []
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden.append(x)
            kvc = past_key_values[i] if past_key_values else None
            x, cache = layer(x, mask=causal_mask,
                              position_ids=position_ids,
                              kv_cache=kvc, use_cache=use_cache)
            if use_cache:
                all_caches.append(cache)

        x      = self.norm(x)
        logits = self.lm_head(x)           # [B, S, vocab_size]

        return {
            "logits":            logits,
            "past_key_values":   all_caches if use_cache else None,
            "hidden_states":     all_hidden if output_hidden_states else None,
            "last_hidden_state": x,
        }

    @torch.no_grad()
    def generate(self, input_ids: Tensor, max_new_tokens: int = 256,
                 temperature: float = 1.0, top_k: int = 50,
                 top_p: float = 0.95, repetition_penalty: float = 1.0,
                 eos_token_id: Optional[int] = None) -> Tensor:
        generated = input_ids
        past      = None
        for _ in range(max_new_tokens):
            out    = self.forward(generated if past is None else generated[:, -1:],
                                  past_key_values=past, use_cache=True)
            logits = out["logits"][:, -1, :] / temperature
            past   = out["past_key_values"]

            # Repetition penalty
            if repetition_penalty != 1.0:
                for tok in set(generated[0].tolist()):
                    logits[0, tok] /= repetition_penalty

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float("-inf")

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                cdf = torch.cumsum(torch.softmax(sorted_logits, -1), dim=-1)
                sorted_logits[cdf - torch.softmax(sorted_logits, -1) > top_p] = float("-inf")
                logits.scatter_(1, sorted_idx, sorted_logits)

            probs  = torch.softmax(logits, dim=-1)
            next_t = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_t], dim=1)

            if eos_token_id is not None and (next_t == eos_token_id).all():
                break

        return generated

"""Text generation utilities for autoregressive language models."""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, List, Callable


def greedy_decode(model, input_ids: Tensor, max_new_tokens: int = 256,
                  eos_token_id: int = 2) -> Tensor:
    """Simple greedy decoding: always pick the highest-probability token."""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(generated)
        logits = (out["logits"] if isinstance(out, dict) else out)[:, -1, :]
        next_tok = logits.argmax(-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
        if (next_tok == eos_token_id).all():
            break
    return generated


def top_k_top_p_filter(logits: Tensor, top_k: int = 0,
                        top_p: float = 1.0, temperature: float = 1.0) -> Tensor:
    """Apply temperature scaling, top-k and top-p (nucleus) filtering."""
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        v, _ = torch.topk(logits, top_k)
        logits[logits < v[:, [-1]]] = float("-inf")
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        cdf = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        remove = cdf - F.softmax(sorted_logits, dim=-1) > top_p
        remove[:, 0] = False
        sorted_logits[remove] = float("-inf")
        logits.scatter_(1, sorted_idx, sorted_logits)
    return logits


def sample_decode(model, input_ids: Tensor, max_new_tokens: int = 256,
                  temperature: float = 1.0, top_k: int = 50,
                  top_p: float = 0.95, repetition_penalty: float = 1.0,
                  eos_token_id: int = 2) -> Tensor:
    """Sampling-based decoding with temperature, top-k, top-p, rep penalty."""
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            out = model(generated)
        logits = (out["logits"] if isinstance(out, dict) else out)[:, -1, :]

        if repetition_penalty != 1.0:
            for tok in set(generated[0].tolist()):
                logits[0, tok] = (logits[0, tok] / repetition_penalty
                                  if logits[0, tok] > 0
                                  else logits[0, tok] * repetition_penalty)

        logits = top_k_top_p_filter(logits, top_k, top_p, temperature)
        probs  = F.softmax(logits, dim=-1)
        next_tok = torch.multinomial(probs, 1)
        generated = torch.cat([generated, next_tok], dim=1)
        if (next_tok == eos_token_id).all():
            break
    return generated


def beam_search(model, input_ids: Tensor, num_beams: int = 4,
                max_new_tokens: int = 256, length_penalty: float = 1.0,
                eos_token_id: int = 2) -> Tensor:
    """Standard beam search decoder."""
    B = input_ids.size(0)
    device = input_ids.device

    # Expand input for num_beams
    beams = input_ids.repeat_interleave(num_beams, dim=0)   # [B*beams, S]
    scores = torch.zeros(B * num_beams, device=device)
    done   = [False] * (B * num_beams)
    finished_beams = [[] for _ in range(B)]

    for step in range(max_new_tokens):
        with torch.no_grad():
            out = model(beams)
        logits = (out["logits"] if isinstance(out, dict) else out)[:, -1, :]
        log_p  = F.log_softmax(logits, dim=-1)

        V = log_p.size(-1)
        # scores: [B*beams, V]
        total = scores.unsqueeze(-1) + log_p
        # Reshape to [B, beams*V] and get top-beams
        total = total.view(B, num_beams * V)
        top_scores, top_idx = torch.topk(total, num_beams, dim=-1)

        beam_idx  = top_idx // V
        token_idx = top_idx % V

        new_beams  = []
        new_scores = []
        for b in range(B):
            for i in range(num_beams):
                bi = b * num_beams + beam_idx[b, i].item()
                ti = token_idx[b, i].item()
                new_seq   = torch.cat([beams[bi], torch.tensor([ti], device=device)])
                new_score = top_scores[b, i].item()
                if ti == eos_token_id:
                    pen = len(new_seq) ** length_penalty
                    finished_beams[b].append((new_score / pen, new_seq))
                else:
                    new_beams.append(new_seq)
                    new_scores.append(new_score)

        if not new_beams:
            break
        max_len = max(s.size(0) for s in new_beams)
        beams   = torch.stack([F.pad(s, (0, max_len-s.size(0))) for s in new_beams])
        scores  = torch.tensor(new_scores, device=device)

    results = []
    for b in range(B):
        if finished_beams[b]:
            best = max(finished_beams[b], key=lambda x: x[0])
            results.append(best[1])
        else:
            results.append(beams[b*num_beams])
    return torch.nn.utils.rnn.pad_sequence(results, batch_first=True)


class StreamingGenerator:
    """Token-by-token streaming generator for real-time output."""
    def __init__(self, model, tokenizer=None):
        self.model     = model
        self.tokenizer = tokenizer

    def stream(self, input_ids: Tensor, max_new_tokens: int = 256,
               temperature: float = 1.0, top_k: int = 50, top_p: float = 0.95):
        """Yields (token_id, decoded_text) pairs as tokens are generated."""
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            with torch.no_grad():
                out    = self.model(generated)
            logits = (out["logits"] if isinstance(out, dict) else out)[:, -1, :]
            logits = top_k_top_p_filter(logits, top_k, top_p, temperature)
            probs  = F.softmax(logits, dim=-1)
            tok    = torch.multinomial(probs, 1)
            generated = torch.cat([generated, tok], dim=1)
            tok_id = tok.item()
            text   = self.tokenizer.decode([tok_id]) if self.tokenizer else str(tok_id)
            yield tok_id, text
            if tok_id == 2:  # EOS
                break

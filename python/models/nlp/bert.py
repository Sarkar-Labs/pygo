"""BERT model for PyGo."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class BERTConfig:
    vocab_size: int=30522; hidden_size: int=768; num_layers: int=12; num_heads: int=12
    intermediate_size: int=3072; max_position_embeddings: int=512; type_vocab_size: int=2
    hidden_dropout: float=0.1; attention_dropout: float=0.1; layer_norm_eps: float=1e-12
    @classmethod
    def base(cls):  return cls()
    @classmethod
    def large(cls): return cls(hidden_size=1024,num_layers=24,num_heads=16,intermediate_size=4096)

class BERTEmbeddings(nn.Module):
    def __init__(self, cfg: BERTConfig):
        super().__init__()
        self.word_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size, padding_idx=0)
        self.pos_emb  = nn.Embedding(cfg.max_position_embeddings, cfg.hidden_size)
        self.type_emb = nn.Embedding(cfg.type_vocab_size, cfg.hidden_size)
        self.ln = nn.LayerNorm(cfg.hidden_size, eps=cfg.layer_norm_eps)
        self.drop = nn.Dropout(cfg.hidden_dropout)
        self.register_buffer("pos_ids", torch.arange(cfg.max_position_embeddings).unsqueeze(0))
    def forward(self, ids, token_type_ids=None):
        S = ids.shape[1]; pos = self.pos_ids[:, :S]
        tt = token_type_ids if token_type_ids is not None else torch.zeros_like(ids)
        return self.drop(self.ln(self.word_emb(ids) + self.pos_emb(pos) + self.type_emb(tt)))

class BERTLayer(nn.Module):
    def __init__(self, cfg: BERTConfig):
        super().__init__(); d = cfg.hidden_size
        self.attn = nn.MultiheadAttention(d, cfg.num_heads, dropout=cfg.attention_dropout, batch_first=True)
        self.attn_ln = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.ff  = nn.Sequential(nn.Linear(d, cfg.intermediate_size), nn.GELU(), nn.Linear(cfg.intermediate_size, d))
        self.ff_ln = nn.LayerNorm(d, eps=cfg.layer_norm_eps)
        self.drop = nn.Dropout(cfg.hidden_dropout)
    def forward(self, x, mask=None):
        a, _ = self.attn(x, x, x, attn_mask=mask)
        x = self.attn_ln(x + self.drop(a))
        return self.ff_ln(x + self.drop(self.ff(x)))

class BERTModel(nn.Module):
    def __init__(self, cfg: BERTConfig = None):
        super().__init__(); self.cfg = cfg or BERTConfig()
        self.embeddings = BERTEmbeddings(self.cfg)
        self.layers = nn.ModuleList([BERTLayer(self.cfg) for _ in range(self.cfg.num_layers)])
        self.pooler = nn.Sequential(nn.Linear(self.cfg.hidden_size, self.cfg.hidden_size), nn.Tanh())
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        x = self.embeddings(input_ids, token_type_ids)
        for layer in self.layers: x = layer(x)
        return {"last_hidden_state": x, "pooler_output": self.pooler(x[:, 0])}

class BERTForSequenceClassification(nn.Module):
    def __init__(self, cfg: BERTConfig = None, num_labels: int = 2):
        super().__init__()
        self.bert = BERTModel(cfg or BERTConfig())
        self.drop = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.cfg.hidden_size, num_labels)
    def forward(self, input_ids, attention_mask=None, labels=None):
        out = self.bert(input_ids, attention_mask)
        logits = self.classifier(self.drop(out["pooler_output"]))
        loss = F.cross_entropy(logits, labels) if labels is not None else None
        return {"logits": logits, "loss": loss}

def bert_base():  return BERTModel(BERTConfig.base())
def bert_large(): return BERTModel(BERTConfig.large())
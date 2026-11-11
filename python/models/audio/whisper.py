"""Whisper ASR model — Radford et al. 2022."""
import torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor
from dataclasses import dataclass
from typing import Optional

@dataclass
class WhisperConfig:
    n_mels:int=80; n_ctx:int=1500; n_state:int=512; n_head:int=8
    n_audio_layer:int=6; n_text_layer:int=6; n_vocab:int=51865; n_text_ctx:int=448

    @classmethod
    def tiny(cls):   return cls(n_state=384,n_head=6,n_audio_layer=4,n_text_layer=4)
    @classmethod
    def base(cls):   return cls(n_state=512,n_head=8,n_audio_layer=6,n_text_layer=6)
    @classmethod
    def small(cls):  return cls(n_state=768,n_head=12,n_audio_layer=12,n_text_layer=12)
    @classmethod
    def medium(cls): return cls(n_state=1024,n_head=16,n_audio_layer=24,n_text_layer=24)
    @classmethod
    def large(cls):  return cls(n_state=1280,n_head=20,n_audio_layer=32,n_text_layer=32)

class Attention(nn.Module):
    def __init__(self,d,n):
        super().__init__(); self.nh=n; self.hd=d//n; self.scale=self.hd**-0.5
        self.q=nn.Linear(d,d); self.k=nn.Linear(d,d,bias=False); self.v=nn.Linear(d,d); self.out=nn.Linear(d,d)
    def forward(self,x,xa=None,mask=None,kv=None):
        B,S,D=x.shape; H,Hd=self.nh,self.hd
        q=self.q(x).view(B,S,H,Hd).transpose(1,2)
        src=xa if xa is not None else x
        k=self.k(src).view(B,-1,H,Hd).transpose(1,2); v=self.v(src).view(B,-1,H,Hd).transpose(1,2)
        if kv is not None and xa is None:
            if "k" in kv: k=torch.cat([kv["k"],k],dim=2); v=torch.cat([kv["v"],v],dim=2)
            kv["k"]=k; kv["v"]=v
        a=q@k.transpose(-2,-1)*self.scale
        if mask is not None: a=a+mask
        return self.out((F.softmax(a,dim=-1)@v).transpose(1,2).reshape(B,S,D))

class EncLayer(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.n_state
        self.attn=Attention(d,cfg.n_head); self.ln1=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d)); self.ln2=nn.LayerNorm(d)
    def forward(self,x): x=x+self.attn(self.ln1(x)); return x+self.ff(self.ln2(x))

class DecLayer(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.n_state
        self.sa=Attention(d,cfg.n_head); self.ln1=nn.LayerNorm(d)
        self.ca=Attention(d,cfg.n_head); self.ln2=nn.LayerNorm(d)
        self.ff=nn.Sequential(nn.Linear(d,d*4),nn.GELU(),nn.Linear(d*4,d)); self.ln3=nn.LayerNorm(d)
    def forward(self,x,xa,mask=None,kv=None):
        x=x+self.sa(self.ln1(x),mask=mask,kv=kv); x=x+self.ca(self.ln2(x),xa=xa); return x+self.ff(self.ln3(x))

class WhisperEncoder(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.n_state
        self.c1=nn.Conv1d(cfg.n_mels,d,3,padding=1); self.c2=nn.Conv1d(d,d,3,stride=2,padding=1)
        self.pe=nn.Embedding(cfg.n_ctx,d); self.layers=nn.ModuleList([EncLayer(cfg) for _ in range(cfg.n_audio_layer)])
        self.ln=nn.LayerNorm(d); self.register_buffer("pids",torch.arange(cfg.n_ctx))
    def forward(self,mel):
        x=F.gelu(self.c1(mel)); x=F.gelu(self.c2(x)).transpose(1,2)
        x=x+self.pe(self.pids[:x.size(1)])
        for l in self.layers: x=l(x)
        return self.ln(x)

class WhisperDecoder(nn.Module):
    def __init__(self,cfg):
        super().__init__(); d=cfg.n_state
        self.te=nn.Embedding(cfg.n_vocab,d); self.pe=nn.Embedding(cfg.n_text_ctx,d)
        self.layers=nn.ModuleList([DecLayer(cfg) for _ in range(cfg.n_text_layer)])
        self.ln=nn.LayerNorm(d); self.register_buffer("pids",torch.arange(cfg.n_text_ctx))
        self.register_buffer("mask",torch.triu(torch.full((cfg.n_text_ctx,cfg.n_text_ctx),float("-inf")),diagonal=1))
    def forward(self,toks,xa,kv=None):
        B,S=toks.shape; off=kv[0]["k"].size(2) if kv and kv[0].get("k") is not None else 0
        x=self.te(toks)+self.pe(self.pids[off:off+S]); m=self.mask[:S,:S+off]
        for i,l in enumerate(self.layers): x=l(x,xa,m,kv[i] if kv else None)
        return self.ln(x)@self.te.weight.T

class Whisper(nn.Module):
    def __init__(self,cfg=None):
        super().__init__(); self.cfg=cfg or WhisperConfig()
        self.encoder=WhisperEncoder(self.cfg); self.decoder=WhisperDecoder(self.cfg)
    def forward(self,mel,toks): return self.decoder(toks,self.encoder(mel))
    @torch.no_grad()
    def transcribe(self,mel,max_len=224,sot=50258,eot=50257):
        xa=self.encoder(mel); toks=torch.tensor([[sot]],device=mel.device)
        for _ in range(max_len):
            next_tok=self.decoder(toks,xa)[:,-1].argmax(-1,keepdim=True)
            toks=torch.cat([toks,next_tok],dim=1)
            if (next_tok==eot).all(): break
        return toks

def whisper_tiny():   return Whisper(WhisperConfig.tiny())
def whisper_base():   return Whisper(WhisperConfig.base())
def whisper_small():  return Whisper(WhisperConfig.small())
def whisper_large():  return Whisper(WhisperConfig.large())

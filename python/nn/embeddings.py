"""Token, position, RoPE, ALiBi, patch embeddings for PyGo."""
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch import Tensor

class TokenEmbedding(nn.Module):
    def __init__(self, vocab, d, pad=0, scale=True):
        super().__init__(); self.scale=math.sqrt(d) if scale else 1.
        self.embed=nn.Embedding(vocab,d,padding_idx=pad)
        nn.init.normal_(self.embed.weight,std=d**-0.5)
        if pad is not None: nn.init.zeros_(self.embed.weight[pad])
    def forward(self,x): return self.embed(x)*self.scale
    @property
    def weight(self): return self.embed.weight

class LearnedPE(nn.Module):
    def __init__(self, max_len, d, dropout=0.):
        super().__init__(); self.embed=nn.Embedding(max_len,d); self.drop=nn.Dropout(dropout)
        nn.init.normal_(self.embed.weight,std=0.01)
        self.register_buffer("pos",torch.arange(max_len).unsqueeze(0))
    def forward(self,x,offset=0): return self.drop(x+self.embed(self.pos[:,offset:offset+x.size(1)]))

class SinusoidalPE(nn.Module):
    def __init__(self,d,max_len=4096,dropout=0.):
        super().__init__(); self.drop=nn.Dropout(dropout)
        pe=torch.zeros(max_len,d); pos=torch.arange(max_len).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.)/d))
        pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div[:d//2])
        self.register_buffer("pe",pe.unsqueeze(0))
    def forward(self,x): return self.drop(x+self.pe[:,:x.size(1)])

class ALiBiBias(nn.Module):
    def __init__(self,n_heads):
        super().__init__()
        slopes=torch.tensor([2**(-8*(h+1)/n_heads) for h in range(n_heads)])
        self.register_buffer("slopes",slopes)
    def forward(self,seq_len):
        pos=torch.arange(seq_len,device=self.slopes.device)
        bias=(pos.unsqueeze(0)-pos.unsqueeze(1)).unsqueeze(0)
        return (bias*self.slopes.view(-1,1,1)).unsqueeze(0)

class PatchEmbedding(nn.Module):
    def __init__(self,img=224,patch=16,in_ch=3,d=768,cls=True):
        super().__init__(); n=(img//patch)**2; self.add_cls=cls
        self.proj=nn.Conv2d(in_ch,d,patch,stride=patch)
        self.cls_token=nn.Parameter(torch.zeros(1,1,d)) if cls else None
        self.pos_embed=nn.Parameter(torch.zeros(1,n+(1 if cls else 0),d))
        self.norm=nn.LayerNorm(d)
        nn.init.trunc_normal_(self.pos_embed,std=0.02)
    def forward(self,x):
        B=x.shape[0]; x=self.proj(x).flatten(2).transpose(1,2)
        if self.add_cls: x=torch.cat([self.cls_token.expand(B,-1,-1),x],dim=1)
        return self.norm(x+self.pos_embed[:,:x.size(1)])
